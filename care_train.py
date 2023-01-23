import dacapo
import logging
import math
import torch
from torchsummary import summary

# CARE task specific elements
from dacapo.experiments.datasplits.datasets.arrays import ZarrArrayConfig, IntensitiesArrayConfig
from dacapo.experiments.datasplits.datasets import RawGTDatasetConfig
from dacapo.experiments.datasplits import TrainValidateDataSplitConfig
from dacapo.experiments.architectures import CNNectomeUNetConfig
from dacapo.experiments.tasks import CARETaskConfig

from dacapo.experiments.trainers import GunpowderTrainerConfig
from dacapo.experiments.trainers.gp_augments import (
    SimpleAugmentConfig,
    ElasticAugmentConfig,
    IntensityAugmentConfig,
)
from funlib.geometry import Coordinate
from dacapo.experiments.run_config import RunConfig
from dacapo.experiments.run import Run
from dacapo.store.create_store import create_config_store
from dacapo.train import train


# set basic login configs
logging.basicConfig(level=logging.INFO)

raw_array_config_zarr = ZarrArrayConfig(
    name="raw",
    file_name="/n/groups/htem/users/br128/data/CBvBottom/CBxs_lobV_bottomp100um_training_0.n5",
    dataset="volumes/raw_30nm",
)

gt_array_config_zarr = ZarrArrayConfig(
    name="gt",
    file_name="/n/groups/htem/users/br128/data/CBvBottom/CBxs_lobV_bottomp100um_training_0.n5",
    dataset="volumes/interpolated_90nm_aligned",
)

raw_array_config_int = IntensitiesArrayConfig(
    name="raw_norm",
    source_array_config = raw_array_config_zarr,
    min = 0.,
    max = 1.
)

gt_array_config_int = IntensitiesArrayConfig(
    name="gt_norm",
    source_array_config = gt_array_config_zarr,
    min = 0.,
    max = 1.
)

dataset_config = RawGTDatasetConfig(
    name="CBxs_lobV_bottomp100um_CARE_0",
    raw_config=raw_array_config_int,
    gt_config=gt_array_config_int,
)

# TODO: check datasplit config, this honestly might work
datasplit_config = TrainValidateDataSplitConfig(
    name="CBxs_lobV_bottomp100um_training_0.n5",
    train_configs=[dataset_config],
    validate_configs=[dataset_config],
)
"""
kernel size 3
2 conv passes per block

1 -- 100%, lose 4 pix - 286 pix
2 -- 50%, lose 8 pix - 142 pix
3 -- 25%, lose 16 pix - 32 pix
"""
# UNET config
architecture_config = CNNectomeUNetConfig(
    name="small_unet",
    input_shape=Coordinate(156, 156, 156),
    # eval_shape_increase=Coordinate(72, 72, 72),
    fmaps_in=1,
    num_fmaps=8,
    fmaps_out=32,
    fmap_inc_factor=4,
    downsample_factors=[(2, 2, 2), (2, 2, 2), (2, 2, 2)],
    constant_upsample=True,
)


# CARE task
task_config = CARETaskConfig(name="CAREModel", num_channels=1, dims=3)


# trainier
trainer_config = GunpowderTrainerConfig(
    name="gunpowder",
    batch_size=2,
    learning_rate=0.0001,
    augments=[
        SimpleAugmentConfig(),
        ElasticAugmentConfig(
            control_point_spacing=(100, 100, 100),
            control_point_displacement_sigma=(10.0, 10.0, 10.0),
            rotation_interval=(0, math.pi / 2.0),
            subsample=8,
            uniform_3d_rotation=True,
        ),
        IntensityAugmentConfig(
            scale=(0.25, 1.75),
            shift=(-0.5, 0.35),
            clip=False,
        ),
    ],
    num_data_fetchers=20,
    snapshot_interval=10000,
    min_masked=0.15,
)


# run config
run_config = RunConfig(
    name="CARE_train",
    task_config=task_config,
    architecture_config=architecture_config,
    trainer_config=trainer_config,
    datasplit_config=datasplit_config,
    repetition=0,
    num_iterations=100000,
    validation_interval=1000,
)

run = Run(run_config)

# run summary TODO create issue
print(summary(run.model, (1, 156, 156, 156)))


# store configs, then train
config_store = create_config_store()

config_store.store_datasplit_config(datasplit_config)
config_store.store_architecture_config(architecture_config)
config_store.store_task_config(task_config)
config_store.store_trainer_config(trainer_config)
config_store.store_run_config(run_config)

# Optional start training by config name:
train(run_config.name)

# CLI dacapo train -r {run_config.name}


""" 
RuntimeError: Can not downsample shape torch.Size([1, 128, 47, 47, 47]) with factor (2, 2, 2), mismatch in spatial dimension 2
"""