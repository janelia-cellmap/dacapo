import os
from pathlib import Path
from typing import Iterable
from dacapo.experiments.datasplits.datasets.arrays import ZarrArrayConfig
from dacapo.experiments.datasplits.datasets import RawGTDatasetConfig
from dacapo.experiments.datasplits import TrainValidateDataSplitConfig


def split_path(path: str):  # TODO add to dacapo.utils
    if ".zarr" in path:
        parts = path.split(".zarr")
        store_type = "zarr"
    elif ".n5" in path:
        parts = path.split(".n5")
        store_type = "n5"
    else:  # TODO add support for other file types
        raise ValueError(f"Invalid file type specification: {path}")
    assert len(parts) == 2, f"Invalid path specification: {path}"
    return parts[0] + f".{store_type}", parts[1].rstrip(os.sep).lstrip(os.sep)


class DataConfig:
    def __init__(self, datasets: Iterable[dict]):
        self.train_configs = []
        self.validate_configs = []
        for i, dataset in enumerate(datasets):  # TODO add support for other file types
            raw_path, raw_dataset = split_path(dataset["raw_path"])
            raw_config = ZarrArrayConfig(
                name=f"{{experiment}}_raw_{i}",
                file_name=Path(raw_path),
                dataset=raw_dataset,
            )
            gt_path, gt_dataset = split_path(dataset["gt_path"])
            gt_config = ZarrArrayConfig(
                name=f"{{experiment}}_gt_{i}",
                file_name=Path(gt_path),
                dataset=gt_dataset,
            )
            if "mask_path" in dataset:
                mask_path, mask_dataset = split_path(dataset["mask_path"])
                mask_config = ZarrArrayConfig(
                    name=f"{{experiment}}_mask_{i}",
                    file_name=Path(mask_path),
                    dataset=mask_dataset,
                )
            else:
                mask_config = None
            # TODO sample points

            dataset_config = RawGTDatasetConfig(
                name=f"{{experiment}}_dataset_{i}",
                raw_config=raw_config,
                gt_config=gt_config,
                mask_config=mask_config,
            )

            if "train" in dataset["role"].lower():
                self.train_configs.append(dataset_config)
            elif "validate" in dataset["role"].lower():
                self.validate_configs.append(dataset_config)
            else:
                raise ValueError(f"Invalid role specification: {dataset['role']}")

        self.datasplit_config = TrainValidateDataSplitConfig(
            name="{experiment}_datasplit",
            train_configs=self.train_configs,
            validate_configs=self.validate_configs,
        )
