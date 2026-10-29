"""Epoch-boundary FP32-master Adam checkpoints. NPZ + JSON, never pickle.

Restore into the same model/optimizer configuration, before a new forward.
Packed storage is copied in place to preserve parameter and gradient bindings.
"""
import json
import os
from pathlib import Path
import random
import tempfile
import numpy as np
from .tensor import Tensor
from .tensor_functions import tensor_from_numpy
from .optim import Adam
from .packed_adam import FusedAdam
from .training_precision import TrainingPrecision


def _layout(model, optimizer):
    if not isinstance(optimizer, (Adam, FusedAdam)):
        raise ValueError('Checkpoint supports Adam and FusedAdam')
    if isinstance(optimizer, FusedAdam):
        optimizer._validate_bindings()
    named = list(model.named_parameters())
    if {id(p) for _, p in named} != {id(p) for p in optimizer.parameters}:
        raise ValueError('Optimizer must cover exactly the model parameters')
    seen, layout = {}, []
    for name, p in named:
        t = p.value
        if not isinstance(t, Tensor) or t.dtype != 'float32' or not t.is_leaf():
            raise ValueError('Checkpoint requires FP32 leaf master parameters')
        alias = seen.setdefault(id(t), name)
        layout.append(dict(name=name, shape=list(t.shape), alias=alias))
    return named, layout


def _moments(optimizer, p):
    if isinstance(optimizer, FusedAdam):
        e = next(e for e in optimizer.entries if e['tensor'] is p.value)
        lo, hi = e['offset'], e['offset'] + p.value.size
        return e['step'], [Tensor.make(s[lo:hi], p.value.shape, backend=p.value.backend)
                           for s in (optimizer.exp_avg, optimizer.exp_avg_sq)]
    s = optimizer._states[id(p)]
    return s.get('step', 0), [s.get('exp_avg'), s.get('exp_avg_sq')]


def read_metadata(path):
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(str(archive['metadata'].item()))
    if meta.get('version') != 1:
        raise ValueError('Unsupported checkpoint version')
    return meta


def save_checkpoint(path, model, optimizer, precision, *, epoch, global_step, config,
                    assets=None):
    named, layout = _layout(model, optimizer)
    if type(epoch) is not int or epoch < 0 or global_step != precision.successful_steps + precision.skipped_steps:
        raise ValueError('Invalid checkpoint progress')
    arrays, steps = {}, []
    for i, (_, p) in enumerate(named):
        arrays[f'p{i}'] = p.value.to_numpy().astype(np.float32)
        step, moments = _moments(optimizer, p)
        steps.append(step)
        for j, moment in enumerate(moments):
            arrays[f'm{i}_{j}'] = (moment.to_numpy().astype(np.float32) if moment is not None
                                    else np.zeros(p.value.shape, np.float32))
    if any(not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError('Cannot save nonfinite parameters or optimizer moments')
    rng = np.random.get_state()
    meta = dict(version=1, layout=layout, optimizer=type(optimizer).__name__, steps=steps,
                hyperparameters={k: getattr(optimizer, k) for k in ('lr', 'beta1', 'beta2', 'eps')},
                precision=precision.state_dict(), epoch=epoch, global_step=global_step,
                config=config, assets=assets or {}, python_rng=random.getstate(),
                numpy_rng=[rng[0], rng[1].tolist(), rng[2], rng[3], rng[4]])
    arrays['metadata'] = np.asarray(json.dumps(meta, allow_nan=False))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Complete the archive before publishing it; refuse to replace existing checkpoints.
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix='.npz.tmp')
    try:
        with os.fdopen(fd, 'wb') as stream:
            np.savez(stream, **arrays)
            stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def load_checkpoint(path, model, optimizer, precision, *, expected_config=None):
    named, layout = _layout(model, optimizer)
    meta = read_metadata(path)
    if (meta['layout'] != layout or meta['optimizer'] != type(optimizer).__name__ or
        meta['hyperparameters'] != {k: getattr(optimizer, k) for k in ('lr', 'beta1', 'beta2', 'eps')}):
        raise ValueError('Checkpoint model layout or optimizer configuration mismatch')
    if expected_config is not None and meta['config'] != expected_config:
        raise ValueError('Checkpoint training configuration/dataset mismatch')
    probe = TrainingPrecision(precision.precision, 'device')
    probe.load_state_dict(meta['precision'])
    if (type(meta['epoch']) is not int or meta['epoch'] < 0 or
        type(meta['global_step']) is not int or
        meta['global_step'] != probe.successful_steps + probe.skipped_steps or
        len(meta['steps']) != len(named) or
        any(type(s) is not int or not 0 <= s <= probe.successful_steps for s in meta['steps'])):
        raise ValueError('Invalid checkpoint progress')
    py_rng = (meta['python_rng'][0], tuple(meta['python_rng'][1]), meta['python_rng'][2])
    nr = meta['numpy_rng']
    np_rng = (nr[0], np.asarray(nr[1], dtype=np.uint32), nr[2], nr[3], nr[4])
    random.Random().setstate(py_rng)
    np.random.RandomState().set_state(np_rng)
    # Validate all arrays before changing any model/optimizer/RNG state.
    values = []
    with np.load(path, allow_pickle=False) as archive:
        for i, (_, p) in enumerate(named):
            group = [archive[key].copy() for key in (f'p{i}', f'm{i}_0', f'm{i}_1')]
            if any(a.shape != p.value.shape or a.dtype != np.float32 or not np.isfinite(a).all() for a in group):
                raise ValueError('Invalid checkpoint tensor shape/dtype/value')
            if np.any(group[2] < 0):
                raise ValueError('Adam second moment must be nonnegative')
            values.append(group)
    alias_values = {}
    for i, spec in enumerate(layout):
        if spec['alias'] in alias_values:
            other = alias_values[spec['alias']]
            if meta['steps'][i] != meta['steps'][other] or any(
                    not np.array_equal(a, b) for a, b in zip(values[i], values[other])):
                raise ValueError('Inconsistent tied parameter state')
        else:
            alias_values[spec['alias']] = i
    for i, (_, p) in enumerate(named):
        t = p.value
        source = tensor_from_numpy(values[i][0], t.backend)
        t.backend.id_map(source, t)
        if t.backend.device_storage:
            t._tensor._storage.owner.version += 1
        if isinstance(optimizer, FusedAdam):
            e = next(e for e in optimizer.entries if e['tensor'] is t)
            _, moments = _moments(optimizer, p)
            for a, target in zip(values[i][1:], moments):
                t.backend.id_map(tensor_from_numpy(a, t.backend), target)
            e['step'] = meta['steps'][i]
        else:
            optimizer._states[id(p)] = (dict(step=meta['steps'][i],
                exp_avg=tensor_from_numpy(values[i][1], t.backend),
                exp_avg_sq=tensor_from_numpy(values[i][2], t.backend)) if meta['steps'][i] else {})
    optimizer.zero_grad()
    precision.load_state_dict(meta['precision'])
    random.setstate(py_rng); np.random.set_state(np_rng)
    return meta
