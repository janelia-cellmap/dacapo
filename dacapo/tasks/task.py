import attr
from funlib.geometry import Coordinate

from dacapo.tasks.predictors import AnyPredictor
from dacapo.tasks.losses import AnyLoss
from dacapo.tasks.post_processors import AnyPostProcessor
from dacapo.converter import converter
from dacapo.tasks.augments import AnyAugment

from typing import List, Optional


@attr.s
class Task:
    name: str = attr.ib(
        metadata={
            "help_text": "Name of your task. This will be saved "
            "so you can find and reuse this task"
        }
    )
    predictors: List[AnyPredictor] = attr.ib(
        metadata={
            "help_text": "The list of predictors for your task. "
            "This defines what outputs you will get."
        }
    )
    losses: List[AnyLoss] = attr.ib(
        metadata={"help_text": "A list of losses, one per predictor."}
    )
    post_processors: List[AnyPostProcessor] = attr.ib(
        metadata={
            "help_text": "A list of postprocessors. One per predictor. "
            "This defines how your model outputs will be turned into a final prediction"
        }
    )
    padding: Optional[Coordinate] = attr.ib(
        default=None, metadata={"help_text": "Any extra padding you may want to add"}
    )
    augments: List[AnyAugment] = attr.ib(
        factory=lambda: list(),
        metadata={"help_text": "Augmentations to apply during training"},
    )

    def verify(self):
        unstructured = converter.unstructure(self)
        structured = converter.structure(unstructured, self.__class__)
        assert self == structured
        return True


converter.register_unstructure_hook(
    List[AnyPredictor],
    lambda o: [{**converter.unstructure(e, unstructure_as=AnyPredictor)} for e in o],
)
converter.register_unstructure_hook(
    List[AnyLoss],
    lambda o: [{**converter.unstructure(e, unstructure_as=AnyLoss)} for e in o],
)
converter.register_unstructure_hook(
    List[AnyPostProcessor],
    lambda o: [
        {**converter.unstructure(e, unstructure_as=AnyPostProcessor)} for e in o
    ],
)