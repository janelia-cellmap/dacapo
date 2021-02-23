from .adam import AdamConfig
from .radam import RAdamConfig

from typing import Union

AnyOptimizer = Union[AdamConfig, RAdamConfig]
