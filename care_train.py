import dacapo
import logging

# CARE task specific elements
from dacapo.datasplits import TrainValidateDataSplitConfig
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

# TODO: check datasplit config, this honestly might work
datasplit_config = TrainValidateDataSplitConfig(
    name="CBxs_lobV_bottomp100um_training_0.n5",
    train_configs = ['/n/groups/htem/users/br128/data/CBvBottom/CBxs_lobV_bottomp100um_training_0.n5/volumes/raw_30nm'],
    validate_configs = ['/n/groups/htem/users/br128/data/CBvBottom/CBxs_lobV_bottomp100um_training_0.n5/volumes/interpolated_90nm_aligned']
)


# UNET config
architecture_config = CNNectomeUNetConfig(
    name="small_unet",
    input_shape=Coordinate(216, 216, 216),
    eval_shape_increase=Coordinate(72, 72, 72),
    fmaps_in=1,
    num_fmaps=8,
    fmaps_out=32,
    fmap_inc_factor=4,
    downsample_factors=[(2, 2, 2), (2, 2, 2), (2, 2, 2)],
    constant_upsample=True,
)


# CARE task
task_config = CARETaskConfig(
    name="CAREModel",
    num_channels=2
)


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
        )
    ],
    num_data_fetchers=20,
    snapshot_interval=10000,
    min_masked=0.15,
    min_labelled=0.1,
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

# run summary
print(torch.summary(run.model, (1, 216, 216, 216)))


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