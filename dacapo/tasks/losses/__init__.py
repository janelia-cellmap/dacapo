from .cross_entropy_loss import CrossEntropyLoss
from .mse_loss import MSELoss

from typing import Union

AnyLoss = Union[MSELoss, CrossEntropyLoss]
