from .argmax import ArgMax
from .watershed import Watershed

from typing import Union

AnyPostProcessor = Union[ArgMax, Watershed]
