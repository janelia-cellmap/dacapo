from typing import List

import attr
from .evaluators import DummyEvaluator
from .losses import DummyLoss
from .post_processors import ArgmaxPostProcessor
from .predictors import OneHotPredictor
from .task import Task


@attr.s
class OneHotTaskParameters:
    classes: List[str] = attr.ib(
        metadata={"help_text": "The classes corresponding with each id starting from 0"}
    )


class OneHotTask(Task):
    """This is a One Hot prediction task that outputs a probability vector
    of length `c` for each voxel where `c` is the number of classes.
    Each voxel prediction has all positive values an l1 norm equal to 1.

    Post processing is extremely easy, the class of each voxel is
    simply the argmax over the vector of output probabilities.
    """

    def __init__(self, params: OneHotTaskParameters):
        self.predictor = OneHotPredictor(classes=params.classes)
        self.loss = DummyLoss()
        self.post_processor = ArgmaxPostProcessor()
        self.evaluator = DummyEvaluator()
