from functools import partial
from contextlib import nullcontext
import time
import os
import fire
import tqdm
import json
import random
import datasets
import numpy as np
from sacrebleu.metrics import BLEU
from transformers import AutoTokenizer
from tokenizers import ByteLevelBPETokenizer

import sys
sys.path.append("./")

import minitorch
from minitorch import DecoderLM
from minitorch.kernel_config import KernelConfig, configure_kernels, make_backend, make_optimizer, attention_dispatch_counts
from minitorch.profiling import CsvMetrics, measure, synchronize, memory_snapshot
from minitorch.training_precision import TrainingPrecision
from minitorch.checkpoint import save_checkpoint, load_checkpoint, read_metadata
import hashlib


WANDB_API_KEY = ""


def create_wandb_run(mode, project, name, config):
    if mode not in ('online', 'offline', 'disabled'):
        raise ValueError('wandb_mode must be online, offline, or disabled')
    if mode == 'disabled':
        return nullcontext(None)
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError('Install wandb: python -m pip install wandb; '
                           'or use --wandb-mode disabled') from exc
    key = WANDB_API_KEY.strip() or os.environ.get('WANDB_API_KEY', '').strip()
    if mode == 'online' and key:
        wandb.login(key=key)
    return wandb.init(project=project, name=name, mode=mode, config=config)


def get_dataset(dataset_name, model_max_length):
    """
    Load and preprocess IWSLT (de-en) dataset.
    
    Args:
        dataset_name (str): Name of the dataset to load
        model_max_length (int): Maximum sequence length for filtering examples

    Returns:
        tuple: (dataset, src_key, tgt_key) where:
            - dataset: Dictionary with 'train', 'validation', 'test' splits
            - src_key (str): Source language key ('de')
            - tgt_key (str): Target language key ('en')
    """
    dataset = {
        split: datasets.load_dataset(dataset_name, split=split)['translation']
        for split in ['train', 'validation', 'test']
    }
    src_key, tgt_key = 'de', 'en'

    dataset = {
        split: [
            example for example in dataset[split]
            if len(example[src_key].split()) + len(
                example[tgt_key].split()) < model_max_length
        ] for split in dataset.keys()
    }

    dataset['test'] = dataset['test'][:100]  # 6750

    print(json.dumps(
        {'data_size': {split: len(dataset[split]) for split in dataset.keys()}},
        indent=4))

    return dataset, src_key, tgt_key


def get_tokenizer(examples, vocab_size, src_key, tgt_key, workdir):
    """
    Train and save a ByteLevelBPETokenizer on the provided dataset.
    
    Args:
        examples (list): Dataset examples for tokenizer training
        vocab_size (int): Desired vocabulary size
        src_key (str): Source language key in examples
        tgt_key (str): Target language key in examples
        workdir (str): Directory to save tokenizer files

    Returns:
        AutoTokenizer: Trained tokenizer with special tokens
                      (e.g., "<eos_de>", "<eos_en>", "<pad>")
    """
    tokenizer = ByteLevelBPETokenizer()

    # Customized training
    tokenizer.train_from_iterator(
        [[example[src_key], example[tgt_key]] for example in examples],
        vocab_size=vocab_size,
        special_tokens=[f'<eos_{src_key}>', f'<eos_{tgt_key}>', '<pad>'])

    tokenizer.save(f'{workdir}/tokenizer.json')
    json.dump({'model_type': 'gpt2'}, open(f'{workdir}/config.json', 'w'))

    tokenizer = AutoTokenizer.from_pretrained(
        workdir,
        eos_token=None,
        bos_token=None,
        pad_token=None,
        unk_token=None)

    return tokenizer


def collate_batch(
        examples, src_key, tgt_key, tokenizer, model_max_length, backend):
    """
    Prepare a batch of examples for model training or evaluation.
    
    Args:
        examples (list): List of examples to process
        src_key (str): Key for source texts in examples
        tgt_key (str): Key for target texts in examples
        tokenizer (AutoTokenizer): Tokenizer for encoding texts
        model_max_length (int): Maximum sequence length
        backend (TensorBackend): Backend for minitorch tensors

    Returns:
        dict: Dictionary containing:
            - input_ids: Tokenized input sequences of shape (batch_size, model_max_length-1)
            - labels: Target sequences of shape (batch_size, model_max_length-1)
            - label_token_weights: Weight mask for loss computation of shape (batch_size, model_max_length-1)
            
    Note:
        input_ids format: <de_tokens> + <de_eos> + <en_tokens> + <en_eos> + <pad>
        labels: Next tokens to predict (shifted by 1)
        label_token_weights: 0 for source tokens, 1 for target tokens
    """
    token_ids, tgt_token_mask = [], []
    max_length = model_max_length
    pad_token_id = tokenizer.vocab['<pad>']
    for example in examples:
        token_ids_src = tokenizer(
            f'{example[src_key]}<eos_{src_key}>')['input_ids']
        token_ids_tgt = tokenizer(
            f'{example[tgt_key]}<eos_{tgt_key}>')['input_ids']

        example_token_ids = token_ids_src + token_ids_tgt
        example_tgt_token_mask = (
                [0] * len(token_ids_src) + [1] * len(token_ids_tgt))
        example_token_ids = example_token_ids[:max_length]
        example_tgt_token_mask = example_tgt_token_mask[:max_length]
        pad_ids = [pad_token_id] * (max_length - len(example_token_ids))

        token_ids.append(example_token_ids + pad_ids)
        tgt_token_mask.append(example_tgt_token_mask + [0] * len(pad_ids))

    # TODO: make examples in a 1d list, provide shape to initialize minitorch.Tensor
    token_ids = np.array(token_ids)
    tgt_token_mask = np.array(tgt_token_mask)

    input_ids = token_ids[:, :-1]
    labels    = token_ids[:, 1:]
    label_token_weights = tgt_token_mask[:, 1:]

    input_ids = minitorch.tensor_from_numpy(input_ids, backend=backend)
    labels    = minitorch.tensor_from_numpy(labels, backend=backend)
    label_token_weights = minitorch.tensor_from_numpy(label_token_weights, backend=backend)
    
    # input_ids = token_ids[:, :-1].tolist()
    # labels    = token_ids[:, 1:].tolist()
    # label_token_weights = tgt_token_mask[:, 1:].tolist()

    # input_ids = minitorch.tensor(input_ids, backend=backend)
    # labels    = minitorch.tensor(labels, backend=backend)
    # label_token_weights = minitorch.tensor(label_token_weights, backend=backend)

    return {
        'input_ids': input_ids,
        'labels': labels,
        'label_token_weights': label_token_weights
    }


def loss_fn(batch, model):
    """
    Compute MLE loss for a batch of examples.
    
    Args:
        batch (dict): Batch data containing 'input_ids', 'labels', 'label_token_weights'
        model (DecoderLM): Language model for prediction

    Returns:
        Tensor: Average loss across all target tokens
    """

    idx = batch['input_ids']
    # print("getting into loss_fn")
    logits = model(idx=idx)
    # print("finish prediction")
    bs, l, c = logits.shape
    logits = logits.view(bs * l, c)
    targets = batch['labels'].view(bs * l)
    label_token_weights = batch['label_token_weights'].view(bs * l)

    # print("start calculating loss")
    # import pdb
    # pdb.set_trace()
    loss = minitorch.nn.softmax_loss(
        logits=logits,
        target=targets
    )

    return ((loss * label_token_weights).sum() / label_token_weights.sum())


def train(model, optimizer, examples, n_samples, collate_fn, batch_size, desc,
          wandb_run=None, global_step=0, epoch=0, metrics=None, warmup_steps=10,
          precision_policy=None):
    from minitorch.training_precision import TrainingPrecision
    policy = precision_policy or TrainingPrecision()
    model.train()
    # Keep the canonical dataset order so epoch-boundary RNG restore reproduces shuffling.
    examples = list(examples)
    random.shuffle(examples)
    examples = examples[:n_samples]
    backend = model.backend
    for i in (prog_bar := tqdm.trange(
            0, len(examples), batch_size, desc=f'Training ({desc})')):
        synchronize(backend)
        before = memory_snapshot(backend)
        if backend.device_storage:
            from minitorch.cuda_storage import runtime
            runtime().reset_peaks()
        total_start = time.perf_counter()
        batch, collate_seconds = measure(backend, lambda: collate_fn(examples=examples[i:i+batch_size]))
        step_start = time.perf_counter()
        update_launches_before = getattr(optimizer, 'kernel_launches', 0)
        optimizer.zero_grad()
        def forward():
            with policy.context():
                return loss_fn(batch=batch, model=model)
        loss, forward_seconds = measure(backend, forward)
        _, backward_seconds = measure(backend, lambda: policy.backward(loss))
        update_metrics, optimizer_seconds = measure(backend, lambda: policy.step(optimizer))
        batch_seconds = time.perf_counter()-step_start
        total_seconds = time.perf_counter()-total_start
        # Scalar logging is outside the compute timer; its transfer is counted.
        loss_value = float(loss.item())
        after = memory_snapshot(backend)
        global_step += 1
        padded_tokens = int(np.prod(batch['input_ids'].shape))
        row = {
            'global_step': global_step, 'epoch': epoch,
            'is_warmup': int(global_step <= warmup_steps),
            'train/loss': loss_value, 'train/learning_rate': float(optimizer.lr),
            'train/tokens': padded_tokens,
            'train/tokens_per_sec': padded_tokens/batch_seconds,
            'train/forward_seconds': forward_seconds,
            'train/backward_seconds': backward_seconds,
            'train/optimizer_seconds': optimizer_seconds,
            'train/batch_seconds': batch_seconds,
            'train/collate_seconds': collate_seconds,
            'train/end_to_end_seconds': total_seconds,
            'precision': policy.precision,
            **update_metrics,
        }
        for key,value in after.items():
            row['memory/'+key] = value-before.get(key,0) if key in (
                'allocations','frees','h2d_bytes','d2h_bytes','d2d_bytes',
                'pool_hits','pool_misses','oom_retries') else value
        for impl,count in attention_dispatch_counts(model).items():
            row['train/attention_'+impl+'_layers'] = count
        if hasattr(optimizer, 'memory_metrics'):
            for key, value in optimizer.memory_metrics().items():
                row['optimizer/'+key] = value-update_launches_before if key == 'update_kernel_launches' else value
        if metrics is not None:
            metrics.log(row)
        if wandb_run is not None:
            wandb_run.log(row)
        prog_bar.set_postfix(tokens_per_sec=row['train/tokens_per_sec'],loss=loss_value,lr=optimizer.lr)
        del loss, batch
    return global_step


def evaluate_loss(model, examples, batch_size, collate_fn, desc, precision_policy=None):
    """
    Evaluate model loss on provided examples.
    
    Args:
        model (DecoderLM): Model to evaluate
        examples (list): Evaluation dataset examples
        batch_size (int): Number of examples per batch
        collate_fn (callable): Function to collate examples into batches
        desc (str): Description for progress bar

    Returns:
        float: Average loss across all batches
    """
    model.eval()
    losses = []

    for i in (prog_bar := tqdm.trange(
        0, len(examples), batch_size, desc=f'Evaluating ({desc})')):
        batch = collate_fn(examples=examples[i:i + batch_size])
        with minitorch.no_grad(), (precision_policy or TrainingPrecision()).context():
            loss = loss_fn(batch=batch, model=model)

        losses.append(loss.item())
        prog_bar.set_postfix(loss=loss.item())

    return np.mean(losses)


def generate(
    model,
    examples,
    src_key,
    tgt_key,
    tokenizer,
    model_max_length,
    backend,
    desc,
    precision_policy=None,
    use_kv_cache=False,
    inference_metrics=None
):
    """
    Generate target sequences for source sequences using argmax decoding.
    
    Args:
        model (DecoderLM): Model for generation
        examples (list): Dataset examples containing source sequences
        src_key (str): Key for source texts in examples
        tgt_key (str): Key for target texts in examples
        tokenizer (AutoTokenizer): Tokenizer for encoding/decoding
        model_max_length (int): Maximum sequence length
        backend (TensorBackend): Backend for minitorch tensors
        desc (str): Description for progress bar

    Returns:
        list: Generated target sequences
    """

    model.eval()
    gen_sents = []
    from minitorch.kv_cache import greedy_token
    policy = precision_policy or TrainingPrecision()
    for example in tqdm.tqdm(examples, desc=f'Generating {desc}'):
        # Run generation for every single example

        token_ids = tokenizer(f'{example[src_key]}<eos_{src_key}>')['input_ids']
        len_src = len(token_ids)

        if not token_ids or len(token_ids) > model_max_length:
            raise ValueError('Source prompt must contain 1..model_max_length tokens')
        cache, next_ids = None, None
        step = 0
        try:
            with minitorch.no_grad(), policy.context():
                while len(token_ids) <= model_max_length:
                    token_start = time.perf_counter()
                    input_ids = next_ids if use_kv_cache and step else minitorch.tensor([token_ids], backend=backend)
                    def forward():
                        nonlocal cache
                        if use_kv_cache:
                            if step == 0:
                                logits,cache = model.prefill(input_ids)
                                return logits
                            return model.decode_step(input_ids,cache)
                        if isinstance(model,minitorch.DecoderLM):
                            return model(idx=input_ids,last_only=True)
                        return model(idx=input_ids)
                    logits,seconds = measure(backend,forward)
                    selection_start = time.perf_counter()
                    if backend.device_storage:
                        # Custom models may return all positions; select their final row.
                        if logits.shape[1] != 1:
                            from minitorch.kv_cache import last_position
                            logits = last_position(logits)
                        next_ids = greedy_token(logits)
                        gen_id = int(next_ids.item())
                    else:
                        gen_id = int(np.argmax(logits.to_numpy()[0,-1,:]))
                    selection_seconds = time.perf_counter()-selection_start
                    token_total_seconds = time.perf_counter()-token_start
                    if inference_metrics is not None:
                        inference_metrics.log(dict(request=len(gen_sents),step=step,
                            phase='prefill' if step==0 else 'decode',
                            context_length=len(token_ids),seconds=seconds,
                            selection_seconds=selection_seconds,token_total_seconds=token_total_seconds,
                            use_kv_cache=use_kv_cache,precision=policy.precision,
                            cache_bytes=cache.nbytes if cache is not None else 0))
                    step += 1
                    if gen_id == tokenizer.vocab[f'<eos_{tgt_key}>']:
                        break
                    token_ids.append(gen_id)
        finally:
            if cache is not None:
                cache.close()

        gen_sents.append(tokenizer.decode(token_ids[len_src:]))

    return gen_sents


def evaluate_bleu(examples, gen_sents, tgt_key):
    """
    Evaluate BLEU score for generated sentences against target sentences.
    
    Args:
        examples (list): Dataset examples containing target sentences
        gen_sents (list): Generated sentences to evaluate
        tgt_key (str): Key for target texts in examples

    Returns:
        dict: Dictionary containing BLEU score
    """
    return {
        'bleu': BLEU().corpus_score(
            hypotheses=gen_sents,
            references=[[example[tgt_key] for example in examples]]).score
    }


def main(
    dataset_name='bbaaaa/iwslt14-de-en-preprocess',
    model_max_length=40,
    n_epochs=20,
    batch_size=128,
    learning_rate=0.02,
    samples_per_epoch=20000,
    n_vocab=10000,
    n_embd=256,
    seed=11111,
    use_fused_kernel=True,
    wandb_project='minitorch-translation',
    wandb_name=None,
    wandb_mode='online',
    attention_impl=None,
    layernorm_impl=None,
    storage_mode='host',
    memory_mode='default',
    dtype='float32',
    use_kv_cache=False,
    metrics_csv=None,
    warmup_steps=10,
    p_dropout=0.1,
    pool_cache_mb=1024,
    optimizer_impl='adam',
    precision='fp32',
    loss_scale=65536.,
    scale_growth_interval=2000,
    checkpoint_dir=None,
    resume=None
):
    """
    Train and evaluate a decoder-only transformer language model.
    
    Args:
        dataset_name (str): Name of the dataset to use, default 'bbaaaa/iwslt14-de-en-preprocess'
        model_max_length (int): Maximum sequence length, default 40
        n_epochs (int): Number of training epochs, default 20
        batch_size (int): Number of examples per batch, default 128
        learning_rate (float): Learning rate for optimizer, default 0.02
        samples_per_epoch (int): Training samples per epoch, default 20000
        n_vocab (int): Vocabulary size for tokenizer, default 10000
        n_embd (int): Embedding dimension, default 256
        seed (int): Random seed, default 11111
        use_fused_kernel (bool): Enable fused softmax and LayerNorm when supported.
        attention_impl (str): reference, fused_softmax or flash; overrides the legacy flag.
        layernorm_impl (str): reference or fused; overrides the legacy flag.
        storage_mode (str): host for original CUDA wrappers, device for resident tensors.
        use_kv_cache (bool): Use inference-only prefill/decode cache during generation (device + flash).
        memory_mode (str): default, pooled, or planned (pool + graph lifetime release + scratch).
        pool_cache_mb (int): Maximum idle memory cached per resident backend, in MiB.
        optimizer_impl (str): adam (original) or fused_adam (packed, device only).
        precision (str): fp32 (default), bf16, or fp16 compute; master parameters stay FP32.
        loss_scale (float): Initial dynamic FP16 loss scale.
        scale_growth_interval (int): Successful FP16 updates between scale growth.
        checkpoint_dir (str): Optional directory for new epoch checkpoints.
        resume (str): Epoch checkpoint to resume; n_epochs is the total target epoch count.
        metrics_csv (str): Path to a new per-step CSV; existing files are rejected.
        warmup_steps (int): Mark initial steps for exclusion from performance summaries.
        p_dropout (float): Model dropout probability; use 0 for the FlashAttention training path.
        wandb_project (str): W&B project for training curves.
        wandb_name (str): Optional run name.
        wandb_mode (str): online, offline, or disabled.
    """

    policy = TrainingPrecision(precision, storage_mode, loss_scale, scale_growth_interval)
    execution = KernelConfig.resolve(
        use_fused_kernel, attention_impl, layernorm_impl, storage_mode=storage_mode,
        memory_mode=memory_mode, dtype=dtype, use_kv_cache=use_kv_cache, pool_cache_mb=pool_cache_mb,
        optimizer_impl=optimizer_impl)
    if not 0 <= p_dropout < 1:
        raise ValueError('p_dropout must be in [0,1)')
    if execution.use_kv_cache and n_embd // 8 > 128:
        raise ValueError('Cached inference requires head_dim<=128')
    np.random.seed(seed)
    random.seed(seed)

    workdir = f'./workdir_vocab{n_vocab}_lr{learning_rate}_embd{n_embd}'
    os.makedirs(workdir, exist_ok=True)

    backend = make_backend(execution)

    config = {
        'n_vocab': n_vocab,  # vocab_size
        'n_embd': n_embd,  # n_embed
        'n_head': 8,  # n_head
        'n_positions': model_max_length,  # n_ctx == n_positions
        # 'n_layer'     : 4,    # n_layer
        'p_dropout': p_dropout,  # x_pdrop
        'ln_eps': 1e-5,  # layer_norm_epsilon
        'backend': backend
    }

    model = DecoderLM(**config)
    configure_kernels(model, execution)
    print(f'Execution: {execution.to_dict()} (unsupported fused shapes use reference)')
    optimizer = make_optimizer(model.parameters(), execution, lr=learning_rate)

    dataset, src_key, tgt_key = get_dataset(
        dataset_name=dataset_name, model_max_length=model_max_length)

    resume_meta = read_metadata(resume) if resume else None
    tokenizer_dir = workdir
    if resume_meta:
        # Restore the exact vocabulary instead of retraining BPE on resume.
        tokenizer_dir = os.path.join(workdir, f'resume_tokenizer_{time.time_ns()}')
        os.makedirs(tokenizer_dir)
        for filename in ('tokenizer.json', 'config.json'):
            with open(os.path.join(tokenizer_dir, filename), 'x', encoding='utf-8') as f:
                f.write(resume_meta['assets'][filename])
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, eos_token=None,
            bos_token=None, pad_token=None, unk_token=None)
    else:
        tokenizer = get_tokenizer(examples=dataset['train'], vocab_size=config['n_vocab'],
            src_key=src_key, tgt_key=tgt_key, workdir=workdir)
    assets = {}
    for filename in ('tokenizer.json', 'config.json'):
        with open(os.path.join(tokenizer_dir, filename), encoding='utf-8') as f:
            assets[filename] = f.read()

    collate_fn = partial(
        collate_batch,
        src_key=src_key,
        tgt_key=tgt_key,
        tokenizer=tokenizer,
        model_max_length=model_max_length,
        backend=backend)

    run_config = {
        'dataset_name': dataset_name, 'model_max_length': model_max_length,
        'n_epochs': n_epochs, 'batch_size': batch_size,
        'learning_rate': learning_rate, 'samples_per_epoch': samples_per_epoch,
        'n_vocab': n_vocab, 'n_embd': n_embd, 'seed': seed,
        'use_fused_kernel': use_fused_kernel, **execution.to_dict(),
        'warmup_steps': warmup_steps, 'p_dropout': p_dropout,
        'timing_scope': 'synchronized wall time; compute excludes collation and logging',
        'token_count': 'padded input tokens',
        'precision': precision, 'loss_scale': loss_scale,
        'scale_growth_interval': scale_growth_interval,
        'resume': str(resume) if resume else None,
    }
    # Fixed configuration and data content determine whether resume is compatible.
    checkpoint_config = {k: v for k, v in run_config.items()
                         if k not in ('n_epochs', 'warmup_steps', 'resume')}
    digest = hashlib.sha256()
    for split in sorted(dataset):
        digest.update(split.encode())
        for example in dataset[split]:
            digest.update(json.dumps(example, sort_keys=True, ensure_ascii=False).encode())
            digest.update(b'\n')
    checkpoint_config['dataset_sha256'] = digest.hexdigest()
    checkpoint_config['tokenizer_sha256'] = hashlib.sha256(assets['tokenizer.json'].encode()).hexdigest()
    global_step, start_epoch = 0, 0
    if resume:
        state = load_checkpoint(resume, model, optimizer, policy, expected_config=checkpoint_config)
        global_step, start_epoch = state['global_step'], state['epoch']
        if n_epochs <= start_epoch:
            raise ValueError('n_epochs must exceed the completed checkpoint epoch')
    run_name = wandb_name or ('fused' if use_fused_kernel else 'baseline')
    if wandb_name is None and execution.attention_impl == 'flash':
        run_name = 'flash_requested' if p_dropout else 'flash'
    metrics_csv = metrics_csv or f'{workdir}/metrics_{storage_mode}_{time.time_ns()}.csv'
    os.makedirs(os.path.dirname(os.path.abspath(metrics_csv)), exist_ok=True)
    if os.path.exists(metrics_csv):
        raise FileExistsError(metrics_csv)
    with open(str(metrics_csv)+'.config.json', 'x', encoding='utf-8') as f:
        json.dump(run_config, f, indent=2)
    print(f'Raw metrics: {metrics_csv}')
    with CsvMetrics(metrics_csv) as metrics, create_wandb_run(wandb_mode, wandb_project, run_name, run_config) as wandb_run:
        if wandb_run is not None:
            wandb_run.define_metric('global_step')
            wandb_run.define_metric('epoch')
            wandb_run.define_metric('train/*', step_metric='global_step')
            wandb_run.define_metric('amp/*', step_metric='global_step')
            wandb_run.define_metric('validation/*', step_metric='epoch')
            wandb_run.define_metric('test/*', step_metric='epoch')
        for epoch_idx in range(start_epoch, n_epochs):
            desc = f'epoch {epoch_idx} / {n_epochs}'

            global_step = train(
                model=model,
                optimizer=optimizer,
                examples=dataset['train'],
                n_samples=samples_per_epoch,
                batch_size=batch_size,
                collate_fn=collate_fn,
                desc=desc, wandb_run=wandb_run,
                global_step=global_step, epoch=epoch_idx + 1,
                metrics=metrics, warmup_steps=warmup_steps, precision_policy=policy)

            validation_loss = evaluate_loss(
                model=model,
                examples=dataset['validation'],
                batch_size=batch_size,
                collate_fn=collate_fn,
                desc=desc, precision_policy=policy)

            print(f'Epoch {epoch_idx}: Validation Loss = {validation_loss}')

            with CsvMetrics(str(metrics_csv)+f'.inference_epoch{epoch_idx+1}.csv') as inference_metrics:
                gen_sents = generate(
                    model=model, examples=dataset['test'], src_key=src_key, tgt_key=tgt_key,
                    tokenizer=tokenizer, model_max_length=model_max_length, backend=backend,
                    desc=desc, precision_policy=policy, use_kv_cache=execution.use_kv_cache,
                    inference_metrics=inference_metrics)

            gen_examples = []
            for example, gen_sent in zip(dataset['test'], gen_sents):
                gen_examples.append({'example': example, 'gen': gen_sent})
            json.dump(gen_examples, open(
                f'{workdir}/gen_epoch{epoch_idx}.json', 'w'), indent=4)

            eval_scores = evaluate_bleu(
                examples=dataset['test'], gen_sents=gen_sents, tgt_key=tgt_key)
            print(f'Epoch {epoch_idx}: {eval_scores}')

            json.dump(
                {'validation_loss': float(validation_loss), **eval_scores},
                open(f'{workdir}/eval_results_epoch{epoch_idx}.json', 'w'))

            if wandb_run is not None:
                wandb_run.log({
                    'global_step': global_step,
                    'epoch': epoch_idx + 1,
                    'validation/loss': float(validation_loss),
                    'test/bleu': float(eval_scores['bleu']),
                })
            if checkpoint_dir:
                checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{epoch_idx+1:04d}.npz')
                save_checkpoint(checkpoint_path, model, optimizer, policy, epoch=epoch_idx+1,
                    global_step=global_step, config=checkpoint_config, assets=assets)
                print(f'Checkpoint: {checkpoint_path}')


if __name__ == '__main__':
    fire.Fire(main)
