import attr

from dacapo.experiments.datasplits.datasets import DatasetConfig
from dacapo.experiments.tasks.predictors import Predictor
from .trainer_config import TrainerConfig

from funlib.geometry import Roi, Coordinate

from typing import Tuple

import numpy as np
import torch


class GeneratorDataset(torch.utils.data.IterableDataset):
    """
    Helper class to return a torch IterableDataset from a generator
    """

    def __init__(self, generator, *args, **kwargs):
        self.generator = generator
        self.args = args
        self.kwargs = kwargs

    def __iter__(self):
        return self.generator(*self.args, **self.kwargs)


@attr.s
class DummyTrainerConfig(TrainerConfig):
    """
    This is just a dummy trainer config used for testing. None of the
    attributes have any particular meaning. This is just to test the trainer
    and the trainer config.

    Attributes:
        mirror_augment (bool): A boolean value indicating whether to use mirror
            augmentation or not.
    Methods:
        verify(self) -> Tuple[bool, str]: This method verifies the DummyTrainerConfig object.

    """

    dummy_attr: bool = attr.ib(metadata={"help_text": "Dummy attribute."})

    def iterable_dataset(
        self,
        datasets: list[DatasetConfig],
        input_shape: Coordinate,
        output_shape: Coordinate,
        predictor: Predictor | None = None,
    ):
        in_roi = Roi(input_shape * 0, input_shape)
        out_roi = Roi(output_shape * 0, output_shape)
        in_voxel_size = datasets[0].raw.voxel_size
        raw = torch.from_numpy(
            datasets[0].raw[in_roi * in_voxel_size].astype(np.float32)
        )
        out_raw = torch.from_numpy(
            datasets[0].raw[out_roi * in_voxel_size].astype(np.float32)
        )

        def generator():
            while True:
                yield {
                    "raw": raw,
                    "target": out_raw,
                    "weight": torch.ones_like(out_raw),
                }

        return GeneratorDataset(generator)
