"""Shared precision policy for the training CLI and offline benchmarks."""
from .amp import autocast, GradScaler


class TrainingPrecision:
    def __init__(self, precision='fp32', storage_mode='host', init_scale=65536.,
                 growth_interval=2000):
        if precision not in ('fp32', 'bf16', 'fp16'):
            raise ValueError('precision must be fp32, bf16, or fp16')
        if precision != 'fp32' and storage_mode != 'device':
            raise ValueError('Mixed precision requires --storage-mode device')
        self.precision = precision
        self.scaler = (None if precision == 'fp32' else GradScaler(
            enabled=precision == 'fp16', init_scale=init_scale,
            growth_interval=growth_interval))
        self.successful_steps = 0
        self.skipped_steps = 0

    def context(self):
        return autocast(dtype='float16' if self.precision == 'fp16' else 'bfloat16',
                        enabled=self.precision != 'fp32')

    def backward(self, loss):
        (self.scaler.scale(loss) if self.scaler else loss).backward()

    def step(self, optimizer):
        scale = self.scaler.get_scale() if self.scaler else 1.
        if self.scaler:
            applied = self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()
            applied = True
        self.successful_steps += int(applied)
        self.skipped_steps += int(not applied)
        return {'amp/scale_used': scale,
                'amp/scale_next': self.scaler.get_scale() if self.scaler else 1.,
                'amp/overflow': int(not applied),
                'train/update_applied': int(applied),
                'train/optimizer_step': self.successful_steps,
                'amp/skipped_steps': self.skipped_steps}

    def state_dict(self):
        return dict(precision=self.precision, successful_steps=self.successful_steps,
                    skipped_steps=self.skipped_steps,
                    scaler=self.scaler.state_dict() if self.scaler else None)

    def load_state_dict(self, state):
        if state['precision'] != self.precision:
            raise ValueError('Checkpoint precision differs from requested precision')
        for key in ('successful_steps', 'skipped_steps'):
            if type(state[key]) is not int or state[key] < 0:
                raise ValueError('Invalid update counter')
        if self.scaler:
            probe = GradScaler()
            probe.load_state_dict(state['scaler'])
            if (probe.enabled != (self.precision == 'fp16') or
                probe.successful_steps != state['successful_steps'] or
                probe.skipped_steps != state['skipped_steps']):
                raise ValueError('Inconsistent scaler state')
            self.scaler.load_state_dict(state['scaler'])
        elif state['scaler'] is not None or state['skipped_steps']:
            raise ValueError('Unexpected FP32 scaler state')
        self.successful_steps = state['successful_steps']
        self.skipped_steps = state['skipped_steps']
