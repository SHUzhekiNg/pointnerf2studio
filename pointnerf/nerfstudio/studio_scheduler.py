from nerfstudio.engine.schedulers import Scheduler, SchedulerConfig
from dataclasses import dataclass, field
from typing import Type
from torch.optim import Optimizer, lr_scheduler
try:
    from torch.optim.lr_scheduler import LRScheduler
except ImportError:
    # Backwards compatibility for PyTorch 1.x
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler


@dataclass
class PointNerfSchedulerConfig(SchedulerConfig):
    """Config for multi step scheduler where lr decays by gamma every milestone"""

    _target: Type = field(default_factory=lambda: PointNerfScheduler)
    """target class to instantiate"""
    lr_decay_iters: int = 1000000
    """The maximum number of steps."""
    lr_decay_exp: float = 0.1
    """The learning rate decay factor."""


class PointNerfScheduler(Scheduler):
    """Multi step scheduler where lr decays by gamma every milestone"""

    config: PointNerfSchedulerConfig

    def get_scheduler(self, optimizer: Optimizer, lr_init: float) -> LRScheduler:
        def func(step):
            lr_l = pow(self.config.lr_decay_exp, step / self.config.lr_decay_iters)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=func)
        return scheduler