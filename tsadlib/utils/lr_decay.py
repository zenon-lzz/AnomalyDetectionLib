"""
=================================================
@Author: Zenon
@Date: 2025-04-03
@Description: This module implements various learning rate scheduling strategies including:
1. Polynomial decay with warmup

"""
from typing import List

from torch.optim.lr_scheduler import _LRScheduler


class PolynomialDecayLR(_LRScheduler):
    """Polynomial learning rate decay scheduler with warmup.
    
    Implements learning rate scheduling according to:
    1. Linear warmup in initial phase
    2. Polynomial decay after warmup
    
    Formula:
    - Warmup: lr = step_count/warmup_updates * base_lr
    - Decay: lr = (base_lr - end_lr) * (1 - (step-warmup)/(total_steps-warmup))^power + end_lr
    
    Attributes:
        optimizer: Wrapped optimizer
        warmup_steps: Number of warmup steps
        total_steps: Total training steps
        lr: Base learning rate after warmup
        end_lr: Final learning rate
        power: Exponent for polynomial decay
        last_epoch: Last epoch index
        verbose: Print learning rate updates
    """

    def __init__(
            self,
            optimizer,
            warmup_steps: int,
            total_steps: int,
            lr: float,
            end_lr: float,
            power: float,
            last_epoch: int = -1,
            verbose: bool = False,
    ):
        """Initialize polynomial decay scheduler with warmup.
        
        Args:
            optimizer: Wrapped optimizer
            warmup_steps: Steps for linear warmup
            total_steps: Total training steps
            lr: Base learning rate after warmup
            end_lr: Minimum learning rate
            power: Polynomial exponent
            last_epoch: Last completed epoch
            verbose: Print LR updates
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.lr = lr
        self.end_lr = end_lr
        self.power = power
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> List[float]:
        """Compute learning rate for current step.
        
        Returns:
            List of learning rates for each parameter group
        """
        if self._step_count <= self.warmup_steps:
            # Linear warmup phase
            self.warmup_factor = self._step_count / float(self.warmup_steps)
            current_lr = self.warmup_factor * self.lr
        elif self._step_count >= self.total_steps:
            # Minimum learning rate
            current_lr = self.end_lr
        else:
            # Polynomial decay phase
            warmup = self.warmup_steps
            lr_range = self.lr - self.end_lr
            progress = (self._step_count - warmup) / (self.total_steps - warmup)
            decay_factor = (1 - progress) ** self.power
            current_lr = lr_range * decay_factor + self.end_lr

        return [current_lr for _ in self.optimizer.param_groups]

    def _get_closed_form_lr(self) -> List[float]:
        """Closed-form solution for learning rate (not implemented)."""
        raise NotImplementedError("Closed form solution not available for this scheduler")
