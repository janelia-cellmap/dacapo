# %%
import dacapo

# %%
from dacapo.store.create_store import create_config_store

config_store = create_config_store()
# %%
from dacapo.experiments.datasplits import DataSplitGenerator
from funlib.geometry import Coordinate

input_resolution = Coordinate(8, 8, 8)
output_resolution = Coordinate(4, 4, 4)
datasplit_config = DataSplitGenerator.generate_from_csv(
    "/misc/public/dacapo_learnathon/datasplit_csvs/cosem_example.csv",
    input_resolution,
    output_resolution,
).compute()

datasplit = datasplit_config.datasplit_type(datasplit_config)
viewer = datasplit._neuroglancer()
config_store.store_datasplit_config(datasplit_config)
# %%
from dacapo.experiments.tasks import DistanceTaskConfig

task_config = DistanceTaskConfig(
    name="cosem_distance_task_4nm",
    channels=["mito"],
    clip_distance=40.0,
    tol_distance=40.0,
    scale_factor=80.0,
)
config_store.store_task_config(task_config)
# %%
from dacapo.experiments.architectures import CNNectomeUNetConfig

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
# %%
from dacapo.experiments.trainers.gp_augments import (
    ElasticAugmentConfig,
    GammaAugmentConfig,
    IntensityAugmentConfig,
    IntensityScaleShiftAugmentConfig,
)
from dacapo.experiments.trainers import GunpowderTrainerConfig

trainer_config = GunpowderTrainerConfig(
    name="cosem_finetune2",
    batch_size=1,
    learning_rate=0.001,
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
    # min_masked=0.05,
    clip_raw=False,
)
config_store.store_trainer_config(trainer_config)
# %%
from dacapo.experiments import RunConfig
from dacapo.experiments.run import Run

from dacapo.experiments.starts import CosemStartConfig

# We will now download a pretrained cosem model and finetune from that model. It will only have to download the first time it is used.

start_config = CosemStartConfig("setup04", "1820500")
# %%
iterations = 2000
validation_interval = iterations // 2
repetitions = 1
for i in range(repetitions):
    run_config = RunConfig(
        name="cosem_distance_run_4nm_finetune2",
        datasplit_config=datasplit_config,
        task_config=task_config,
        architecture_config=architecture_config,
        trainer_config=trainer_config,
        num_iterations=iterations,
        validation_interval=validation_interval,
        repetition=i,
        start_config=start_config,
    )

    print(run_config.name)
    config_store.store_run_config(run_config)
# %%
from dacapo.train import train_run
from dacapo.experiments.run import Run
from dacapo.store.create_store import create_config_store

config_store = create_config_store()

run = Run(config_store.retrieve_run_config("cosem_distance_run_4nm_finetune"))
# we already trained it, so we will just load the weights
# train_run(run)
# %%

# %%
