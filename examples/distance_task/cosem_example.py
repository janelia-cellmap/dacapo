# %% [markdown]
# # Dacapo
#
# DaCapo is a framework that allows for easy configuration and execution of established machine learning techniques on arbitrarily large volumes of multi-dimensional images.
#
# DaCapo has 4 major configurable components:
# 1. **dacapo.datasplits.DataSplit**
#
# 2. **dacapo.tasks.Task**
#
# 3. **dacapo.architectures.Architecture**
#
# 4. **dacapo.trainers.Trainer**
#
# These are then combined in a single **dacapo.experiments.Run** that includes your starting point (whether you want to start training from scratch or continue off of a previously trained model) and stopping criterion (the number of iterations you want to train).

# %% [markdown]
# ## Environment setup
# If you have not already done so, you will need to install DaCapo. You can do this by first creating a new environment and then installing DaCapo using pip.
#
# ```bash
# conda create -n dacapo python=3.10
# conda activate dacapo
# ```
#
# Then, you can install DaCapo using pip, via GitHub:
#
# ```bash
# pip install git+https://github.com/janelia-cellmap/dacapo.git
# ```
#
# Or you can clone the repository and install it locally:
#
# ```bash
# git clone https://github.com/janelia-cellmap/dacapo.git
# cd dacapo
# pip install -e .
# ```
#
# Be sure to select this environment in your Jupyter notebook or JupyterLab.

# %% [markdown]
"""
## Config Store
To define where the data goes, create a dacapo.yaml configuration file either in `~/.config/dacapo/dacapo.yaml` or in `./dacapo.yaml`. Here is a template:

```yaml
type: files
runs_base_dir: /path/to/my/data/storage
```
The `runs_base_dir` defines where your on-disk data will be stored. The `type` setting determines the database backend. The default is `files`, which stores the data in a file tree on disk. Alternatively, you can use `mongodb` to store the data in a MongoDB database. To use MongoDB, you will need to provide a `mongodbhost` and `mongodbname` in the configuration file:

```yaml
...
mongodbhost: mongodb://dbuser:dbpass@dburl:dbport/
mongodbname: dacapo
"""

# %%
# First we need to create a config store to store our configurations
from dacapo.store.create_store import create_config_store

config_store = create_config_store()


# %% [markdown]
# ## Datasplit
# Where can you find your data? What format is it in? Does it need to be normalized? What data do you want to use for validation?

# We'll assume your data is in a zarr file, and that you have a raw and a ground truth dataset, all stored in your `runs_base_dir` as `example_{type}.zarr` where `{type}` is either `train` or `validate`.
# NOTE: You may need to delete old config stores if you are re-running this cell with modifications to the configs. The config names are unique and will throw an error if you try to store a config with the same name as an existing config. For the `files` backend, you can delete the `runs_base_dir/configs` directory to remove all stored configs.

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

# %% [markdown]
# ## Task
# What do you want to learn? An instance segmentation? If so, how? Affinities,
# Distance Transform, Foreground/Background, etc. Each of these tasks are commonly learned
# and evaluated with specific loss functions and evaluation metrics. Some tasks may
# also require specific non-linearities or output formats from your model.

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
        IntensityAugmentConfig(scale=(0.25, 1.75), shift=(-0.5, 0.35), clip=True),
        GammaAugmentConfig(gamma_range=(0.5, 2.0)),
        IntensityScaleShiftAugmentConfig(scale=2.0, shift=-1.0),
    ],
    snapshot_interval=10000,
    min_masked=0.05,
    clip_raw=False,
)
config_store.store_trainer_config(trainer_config)

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
repetitions = 1
for i in range(repetitions):
    run_config = RunConfig(
        name="scratch_cosem_distance_run_4nm",
        # # NOTE: This is a template for the name of the run. You can customize it as you see fit.
        # name=("_").join(
        #     [
        #         "example",
        #         "scratch" if start_config is None else "finetuned",
        #         datasplit_config.name,
        #         task_config.name,
        #         architecture_config.name,
        #         trainer_config.name,
        #     ]
        # )
        # + f"__{i}",
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

# %% [markdown]
# ## Train

# NOTE: The run stats are stored in the `runs_base_dir/stats` directory. You can delete this directory to remove all stored stats if you want to re-run training. Otherwise, the stats will be appended to the existing files, and the run won't start from scratch. This may cause errors
# %%
from dacapo.train import train_run
from dacapo.experiments.run import Run
from dacapo.store.create_store import create_config_store

config_store = create_config_store()

run = Run(config_store.retrieve_run_config("cosem_distance_run_4nm"))
train_run(run)
