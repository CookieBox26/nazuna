import torch


class WarmupInverseSqrtLR(torch.optim.lr_scheduler.LRScheduler):
    def __init__(
        self, optimizer, warmup_steps: int = 1000,
        start_factor: float = 1e-3, decay_power: float = 0.5,
    ):
        assert warmup_steps > 0
        self.warmup_steps = warmup_steps
        self.start_factor = start_factor
        self.decay_power = decay_power
        super().__init__(optimizer)

    def get_lr(self):
        step = self.last_epoch + 1
        if step < self.warmup_steps:
            factor = self.start_factor + (1.0 - self.start_factor) * step / self.warmup_steps
        else:
            factor = (self.warmup_steps / step) ** self.decay_power
        return [base_lr * factor for base_lr in self.base_lrs]
