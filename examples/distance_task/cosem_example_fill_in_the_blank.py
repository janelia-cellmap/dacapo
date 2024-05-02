# %%
# First we need to create a config store to store our configurations
from dacapo.store.create_store import create_config_store

# create the config store
config_store = ...
# %% [markdown]
# ## Datasplit
# Where can you find your data? What format is it in? Does it need to be normalized? What data do you want to use for validation?

# We'll assume your data is in a zarr file, and that you have a raw and a ground truth dataset, all stored in your `runs_base_dir` as `example_{type}.zarr` where `{type}` is either `train` or `validate`.
# NOTE: You may need to delete old config stores if you are re-running this cell with modifications to the configs. The config names are unique and will throw an error if you try to store a config with the same name as an existing config. For the `files` backend, you can delete the `runs_base_dir/configs` directory to remove all stored configs.

# %%
from dacapo.experiments.datasplits import DataSplitGenerator
from funlib.geometry import Coordinate

# We will be working with cosem data and we want to work with 8nm isotropic input resolution for the raw data and output at 4 nm resolution.
# Create these resolutions as Coordinates.
input_resolution = ...
output_resolution = ...

# Create the datasplit config using the cosem_example.csv located in the shared learnathon examples
datasplit_config = ...

# Create the datasplit, produce the neuroglancer link and store the datasplit
datasplit = ...
viewer = ...
config_store
# %% [markdown]
# ## Task
# What do you want to learn? An instance segmentation? If so, how? Affinities,
# Distance Transform, Foreground/Background, etc. Each of these tasks are commonly learned
# and evaluated with specific loss functions and evaluation metrics. Some tasks may
# also require specific non-linearities or output formats from your model.

# %%
from dacapo.experiments.tasks import DistanceTaskConfig

# Create a distance task config where the clip_distance=tol_distance=10x the output resolution,
# and scale_factor = 20x the output resolution
task_config = ...
config_store
# %% [markdown]
# ## Architecture
#
# The setup of the network you will train. Biomedical image to image translation often utilizes a UNet, but even after choosing a UNet you still need to provide some additional parameters. How much do you want to downsample? How many convolutional layers do you want?

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

# %% [markdown]
# ## Trainer
#
# How do you want to train? This config defines the training loop and how the other three components work together. What sort of augmentations to apply during training, what learning rate and optimizer to use, what batch size to train with.

# %%
from dacapo.experiments.trainers import GunpowderTrainerConfig
from dacapo.experiments.trainers.gp_augments import (
    ElasticAugmentConfig,
    GammaAugmentConfig,
    IntensityAugmentConfig,
    IntensityScaleShiftAugmentConfig,
)

trainer_config = GunpowderTrainerConfig(
    name="cosem",
    batch_size=1,
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
        # Create an intensity augment config scaling from .25 to 1.25, shifting from -.5 to .35, and with clipping
        ...,
        # Create a gamma augment config with range .5 to 2
        ...,
        # Create an intensity scale shift agument config to rescale data from the range 0->1 to -1->1
        ...,
    ],
    snapshot_interval=10000,
    min_masked=0.05,
    clip_raw=False,
)
# Store the trainer
config_store

# %% [markdown]
# ## Run
# Now that we have our components configured, we just need to combine them into a run and start training. We can have multiple repetitions of a single set of configs in order to increase our chances of finding an optimum.

# %%
from dacapo.experiments import RunConfig
from dacapo.experiments.run import Run

start_config = None

# Uncomment to start from a pretrained model
# start_config = StartConfig(
#     "setup04",
#     "best",
# )

iterations = 2000
validation_interval = iterations // 2
#  Set up a run using all of the configs and settings you created above
run_config = ...

print(run_config.name)
config_store

# %% [markdown]
# ## Train

# NOTE: The run stats are stored in the `runs_base_dir/stats` directory. You can delete this directory to remove all stored stats if you want to re-run training. Otherwise, the stats will be appended to the existing files, and the run won't start from scratch. This may cause errors
# %%
from dacapo.train import train_run
from dacapo.experiments.run import Run

# load the run and train it
run = Run(config_store)
train_run(run)
