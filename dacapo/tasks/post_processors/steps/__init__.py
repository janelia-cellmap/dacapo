from .argmax import ArgMaxStep
from .fragment import Fragment
from .agglomerate import Agglomerate
from .create_luts import CreateLuts
from .segment import Segment

from typing import Union

AnyProcessingStep = Union[
    ArgMaxStep,
    Fragment,
    Agglomerate,
    CreateLuts,
    Segment,
]
