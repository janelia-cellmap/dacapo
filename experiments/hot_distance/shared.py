# %%
from dacapo.store.create_store import create_config_store
from dacapo.experiments.datasplits import DataSplitGenerator
from funlib.geometry import Coordinate
from dacapo.experiments.architectures import CNNectomeUNetConfig
from dacapo.experiments.trainers import GunpowderTrainerConfig
from dacapo.experiments.trainers.gp_augments import (
    ElasticAugmentConfig,
    GammaAugmentConfig,
    IntensityAugmentConfig,
    IntensityScaleShiftAugmentConfig,
)
from dacapo.experiments import RunConfig
from dacapo.experiments.run import Run
import subprocess

config_store = create_config_store()

input_resolution = Coordinate(8, 8, 8)
output_resolution = Coordinate(4, 4, 4)

architecture_config = CNNectomeUNetConfig(
    name="upsample_unet",
    input_shape=Coordinate(216, 216, 216),
    eval_shape_increase=Coordinate(72, 72, 72),
    fmaps_in=1,
    num_fmaps=12,
    fmaps_out=72,
    fmap_inc_factor=6,
    downsample_factors=[(2, 2, 2), (3, 3, 3), (3, 3, 3)],
    constant_upsample=True,
    upsample_factors=[(2, 2, 2)],
)
config_store.store_architecture_config(architecture_config)

trainer_config = GunpowderTrainerConfig(
    name="trainer",
    batch_size=4,
    learning_rate=0.0001,
    num_data_fetchers=20,
    augments=[
        ElasticAugmentConfig(
            control_point_spacing=[100, 100, 100],
            control_point_displacement_sigma=[10.0, 10.0, 10.0],
            rotation_interval=(0.0, 1.5707963267948966),
            subsample=8,
            uniform_3d_rotation=True,
        ),
        IntensityAugmentConfig(scale=(0.25, 1.75), shift=(-0.5, 0.35), clip=True),
        GammaAugmentConfig(gamma_range=(0.5, 2.0)),
        IntensityScaleShiftAugmentConfig(scale=2.0, shift=-1.0),
    ],
    snapshot_interval=10000,
    min_masked=0.05,
    clip_raw=False,
)
config_store.store_trainer_config(trainer_config)

iterations = 300000
validation_interval = 10000
repetitions = 3

run_command = 'bsub -n 8 -gpu "num=4" -q gpu_tesla -e "train_{task_type}_$(date +%Y%m%d%H%M%S).err" -o "train_{task_type}_$(date +%Y%m%d%H%M%S).out" python dacapo train {run_name}'
