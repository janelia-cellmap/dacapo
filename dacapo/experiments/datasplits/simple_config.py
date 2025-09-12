from .datasets.simple import SimpleDataset
from .datasplit_config import DataSplitConfig

import attr

from pathlib import Path

import glob


@attr.s
class SimpleDataSplitConfig(DataSplitConfig):
    """
    A convention over configuration datasplit that can handle many of the most
    basic cases.
    """

    path: Path = attr.ib()
    name: str = attr.ib()
    train_group_name: str = attr.ib(default="train")
    validate_group_name: str = attr.ib(default="test")
    raw_name: str = attr.ib(default="raw")
    gt_name: str = attr.ib(default="labels")
    mask_name: str = attr.ib(default="mask")

    @staticmethod
    def datasplit_type(datasplit_config):
        return datasplit_config

    def get_paths(self, group_name: str) -> list[Path]:
        level_0 = f"{self.path}/{self.raw_name}"
        level_1 = f"{self.path}/{group_name}/{self.raw_name}"
        level_2 = f"{self.path}/{group_name}/**/{self.raw_name}"
        level_0_matches = glob.glob(level_0)
        level_1_matches = glob.glob(level_1)
        level_2_matches = glob.glob(level_2)
        if len(level_0_matches) > 0:
            assert (
                len(level_1_matches) == len(level_2_matches) == 0
            ), f"Found raw data at {level_0} and {level_1} and {level_2}"
            return [Path(x).parent for x in level_0_matches]
        elif len(level_1_matches) > 0:
            assert (
                len(level_2_matches) == 0
            ), f"Found raw data at {level_1} and {level_2}"
            return [Path(x).parent for x in level_1_matches]
        elif len(level_2_matches) > 0:
            return [Path(x).parent for x in level_2_matches]

        raise Exception(f"No raw data found at {level_0} or {level_1} or {level_2}")

    @property
    def train(self) -> list[SimpleDataset]:
        return [
            SimpleDataset(
                name=x.stem,
                path=x,
            )
            for x in self.get_paths(self.train_group_name)
        ]

    @property
    def validate(self) -> list[SimpleDataset]:
        return [
            SimpleDataset(
                name=x.stem,
                path=x,
            )
            for x in self.get_paths(self.validate_group_name)
        ]
