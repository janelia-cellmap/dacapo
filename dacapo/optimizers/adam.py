from torch.optim import Adam as AdamOptimizer

from .algorithm_abc import Algorithm

from typing import Tuple
import attr


@attr.s
class Adam(Algorithm):
    lr: float = attr.ib(
        default=0.001,
        metadata={"help_text": "The learning rate."},
    )
    betas: Tuple[float, float] = attr.ib(
        default=(0.9, 0.999),
        metadata={
            "help_text": "coefficients used for computing running averages of "
            "gradient and its square."
        },
    )
    eps: float = attr.ib(
        default=1e-08,
        metadata={
            "help_text": "term added to the denominator to improve numerical stability."
        },
    )
    weight_decay: float = attr.ib(
        default=0,
        metadata={"help_text": "weight decay (L2 penalty)."},
    )
    amsgrad: bool = attr.ib(
        default=False,
        metadata={"help_text": "Whether to use the AMSGrad variant."},
    )

    def instance(self, params):
        return AdamOptimizer(
            params,
            lr=self.lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
            amsgrad=self.amsgrad,
        )
