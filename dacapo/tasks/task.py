from typing import List, Optional
import attr

from funlib.geometry import Coordinate

from dacapo.tasks.predictors import AnyPredictor
from dacapo.tasks.losses import AnyLoss
from dacapo.tasks.post_processors import AnyPostProcessor


@attr.s
class Task:
    name: str = attr.ib()
    predictors: List[AnyPredictor] = attr.ib()
    losses: List[AnyLoss] = attr.ib()
    post_processors: List[AnyPostProcessor] = attr.ib()
    weighting_type: str = attr.ib()
    padding: Optional[Coordinate] = attr.ib()
