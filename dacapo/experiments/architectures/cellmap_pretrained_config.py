import attr
from typing import Iterable, Tuple
from dacapo.experiments.architectures import ArchitectureConfig
import cellmap_models


@attr.s
class CellMapPretrainedConfig(ArchitectureConfig):
    """ """

    name: str = attr.ib(
        metadata={
            "help_text": "A unique name for this architecture. This will be saved so "
            "you and others can find and reuse this task. Keep it short "
            "and avoid special characters."
        }
    )

    model_type: str = attr.ib(
        metadata={
            "help_text": "The type of the pretrained model to be used. Currently only 'cosem' is supported."
        }
    )

    model_name: str = attr.ib(
        metadata={"help_text": "The name of the pretrained model to be used."}
    )

    checkpoint: int = attr.ib(
        metadata={"help_text": "The checkpoint of the pretrained model to be used."}
    )

    input_shape: Iterable[int] = attr.ib(
        default=None,
        metadata={
            "help_text": "The input shape of the model. Defaults to minimum input shape. Will be rounded up to the nearest valid shape or rounded down based on fit_mode = `grow` or `shrink` respectively."
        },
    )

    eval_shape_increase: Iterable[int] = attr.ib(
        default=None,
        metadata={
            "help_text": "The increase in input shape during evaluation. Defaults to 0. Will be rounded up to the nearest valid shape or rounded down based on fit_mode = `grow` or `shrink` respectively"
        },
    )

    fit_mode: str = attr.ib(
        default="shrink",
        metadata={
            "help_text": "The mode to fit the input shape. 'shrink' will reduce the input shape to the nearest valid shape. 'grow' will increase the input shape to the nearest valid shape."
        },
    )

    def verify(self) -> Tuple[bool, str]:
        """
        A method to validate an architecture setup.

        Returns
        -------
        bool
            A flag indicating whether the config is valid or not.
        str
            A description of the architecture.
        """
        # Make sure weights are downloaded and loadable
        try:
            model_loader = getattr(cellmap_models, self.model_type)
            model = model_loader.load(f"{self.model_name}/{self.checkpoint}")
            return True, model.__str__()
        except:  # noqa
            return False, f"Model type {self.model_type} could not be loaded."
