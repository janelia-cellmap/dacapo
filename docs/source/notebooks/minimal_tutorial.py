# %% [markdown]
# # Minimal Tutorial
#

# %% [markdown]
# ## Introduction and overview
#
# In this tutorial we will cover the basics of running an ML experiment with DaCapo.
#
# DaCapo has 4 major configurable components:
#
# 1. **dacapo.datasplits.DataSplit**
#
# 2. **dacapo.tasks.Task**
#
# 3. **dacapo.architectures.Architecture**
#
# 4. **dacapo.trainers.Trainer**
#
# These are then combined in a single **dacapo.experiments.Run** that includes
# your starting point (whether you want to start training from scratch or
# continue off of a previously trained model) and stopping criterion (the number
# of iterations you want to train).

# %% [markdown]
# ## Environment setup
# If you have not already done so, you will need to install DaCapo. You can do this
# by first creating a new environment and then installing DaCapo using pip.
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
# ```bash
# pip install dacapo-ml
# ```
#
# Be sure to select this environment in your Jupyter notebook or JupyterLab.

# %% [markdown]
# ## Config Store
# To define where the data goes, create a dacapo.yaml configuration file either in `~/.config/dacapo/dacapo.yaml` or in `./dacapo.yaml`. Here is a template:
#
# ```yaml
# type: files
# runs_base_dir: /path/to/my/data/storage
# ```
# The `runs_base_dir` defines where your on-disk data will be stored. The `type` setting determines the database backend. The default is `files`, which stores the data in a file tree on disk. Alternatively, you can use `mongodb` to store the data in a MongoDB database. To use MongoDB, you will need to provide a `mongodbhost` and `mongodbname` in the configuration file:
#
# ```yaml
# mongodbhost: mongodb://dbuser:dbpass@dburl:dbport/
# mongodbname: dacapo
# ```

# %%
# First we need to create a config store to store our configurations
import multiprocessing

multiprocessing.set_start_method("fork", force=True)
from dacapo.store.create_store import create_config_store, create_stats_store

config_store = create_config_store()

# %% Create some data

import random

import matplotlib.pyplot as plt
import numpy as np
from funlib.geometry import Coordinate, Roi
from funlib.persistence import prepare_ds
from scipy.ndimage import label
from skimage import data
from skimage.filters import gaussian

from dacapo.utils.affinities import seg_to_affgraph

# Download the data
cell_data = (data.cells3d().transpose((1, 0, 2, 3)) / 256).astype(np.uint8)

# Handle metadata
offset = Coordinate(0, 0, 0)
voxel_size = Coordinate(290, 260, 260)
axis_names = ["c^", "z", "y", "x"]
units = ["nm", "nm", "nm"]

# Create the zarr array with appropriate metadata
cell_array = prepare_ds(
    "cells3d.zarr",
    "raw",
    Roi((0, 0, 0), cell_data.shape[1:]) * voxel_size,
    voxel_size=voxel_size,
    dtype=np.uint8,
    num_channels=None,
)

# Save the cell data to the zarr array
cell_array[cell_array.roi] = cell_data[1]

# Generate and save some pseudo ground truth data
mask_array = prepare_ds(
    "cells3d.zarr",
    "mask",
    Roi((0, 0, 0), cell_data.shape[1:]) * voxel_size,
    voxel_size=voxel_size,
    dtype=np.uint8,
)
cell_mask = np.clip(gaussian(cell_data[1] / 255.0, sigma=1), 0, 255) * 255 > 30
not_membrane_mask = np.clip(gaussian(cell_data[0] / 255.0, sigma=1), 0, 255) * 255 < 10
mask_array[mask_array.roi] = cell_mask * not_membrane_mask

# Generate labels via connected components
labels_array = prepare_ds(
    "cells3d.zarr",
    "labels",
    Roi((0, 0, 0), cell_data.shape[1:]) * voxel_size,
    voxel_size=voxel_size,
    dtype=np.uint8,
)
labels_array[labels_array.roi] = label(mask_array.to_ndarray(mask_array.roi))[0]

# Generate affinity graph
affs_array = prepare_ds(
    "cells3d.zarr",
    "affs",
    Roi((0, 0, 0), cell_data.shape[1:]) * voxel_size,
    voxel_size=voxel_size,
    num_channels=3,
    dtype=np.uint8,
)
affs_array[affs_array.roi] = (
    seg_to_affgraph(
        labels_array.to_ndarray(labels_array.roi),
        neighborhood=[Coordinate(1, 0, 0), Coordinate(0, 1, 0), Coordinate(0, 0, 1)],
    )
    * 255
)


# %% [markdown]
# Here we show a slice of the raw data:
# %%
plt.imshow(cell_array.data[30])

# %% [markdown]
# ## Datasplit
# Where can you find your data? What format is it in? Does it need to be normalized?
# What data do you want to use for validation?

# We have already saved some data in `cells3d.zarr`. We will use this data for
# training and validation. We only have one dataset, so we will be using the
# same data for both training and validation. This is not recommended for real
# experiments, but is useful for this tutorial.

# %%
from dacapo.experiments.datasplits import TrainValidateDataSplitConfig
from dacapo.experiments.datasplits.datasets import RawGTDatasetConfig
from dacapo.experiments.datasplits.datasets.arrays import (
    ZarrArrayConfig,
    IntensitiesArrayConfig,
)
from funlib.geometry import Coordinate

datasplit_config = TrainValidateDataSplitConfig(
    name="example_datasplit",
    train_configs=[
        RawGTDatasetConfig(
            name="example_dataset",
            raw_config=IntensitiesArrayConfig(
                name="example_raw_normalized",
                source_array_config=ZarrArrayConfig(
                    name="example_raw",
                    file_name="cells3d.zarr",
                    dataset="raw",
                ),
                min=0,
                max=255,
            ),
            gt_config=ZarrArrayConfig(
                name="example_gt",
                file_name="cells3d.zarr",
                dataset="mask",
            ),
        )
    ],
    validate_configs=[
        RawGTDatasetConfig(
            name="example_dataset",
            raw_config=IntensitiesArrayConfig(
                name="example_raw_normalized",
                source_array_config=ZarrArrayConfig(
                    name="example_raw",
                    file_name="cells3d.zarr",
                    dataset="raw",
                ),
                min=0,
                max=255,
            ),
            gt_config=ZarrArrayConfig(
                name="example_gt",
                file_name="cells3d.zarr",
                dataset="mask",
            ),
        )
    ],
)

datasplit = datasplit_config.datasplit_type(datasplit_config)
config_store.store_datasplit_config(datasplit_config)

# %% [markdown]
# ## Task
# What do you want to learn? An instance segmentation? If so, how? Affinities,
# Distance Transform, Foreground/Background, etc. Each of these tasks are commonly learned
# and evaluated with specific loss functions and evaluation metrics. Some tasks may
# also require specific non-linearities or output formats from your model.

# %%
from dacapo.experiments.tasks import DistanceTaskConfig, AffinitiesTaskConfig

# an example distance task configuration
# note that the clip_distance, tol_distance, and scale_factor are in nm
dist_task_config = DistanceTaskConfig(
    name="example_dist",
    channels=["mito"],
    clip_distance=260 * 10.0,
    tol_distance=260 * 10.0,
    scale_factor=260 * 20.0,
)
config_store.store_task_config(dist_task_config)

# an example affinities task configuration
affs_task_config = AffinitiesTaskConfig(
    name="example_affs",
    neighborhood=[(0, 1, 0), (0, 0, 1)],
)
config_store.store_task_config(affs_task_config)

# %% [markdown]
# ## Architecture
#
# The setup of the network you will train. Biomedical image to image translation
# often utilizes a UNet, but even after choosing a UNet you still need to provide
# some additional parameters. How much do you want to downsample? How many
# convolutional layers do you want?

# %%
from dacapo.experiments.architectures import CNNectomeUNetConfig

# Note we make this UNet 2D by defining kernel_size_down, kernel_size_up, and downsample_factors
# all with 1s in z meaning no downsampling or convolving in the z direction.
architecture_config = CNNectomeUNetConfig(
    name="example_unet",
    input_shape=(2, 64, 64),
    eval_shape_increase=(7, 0, 0),
    fmaps_in=1,
    num_fmaps=8,
    fmaps_out=8,
    fmap_inc_factor=2,
    downsample_factors=[(1, 4, 4), (1, 4, 4)],
    kernel_size_down=[[(1, 3, 3)] * 2] * 3,
    kernel_size_up=[[(1, 3, 3)] * 2] * 2,
    constant_upsample=True,
    padding="same",
)
config_store.store_architecture_config(architecture_config)

# %% [markdown]
# ## Trainer
#
# How do you want to train? This config defines the training loop and how
# the other three components work together. What sort of augmentations to
# apply during training, what learning rate and optimizer to use, what
# batch size to train with.

# %%
from dacapo.experiments.trainers import GunpowderTrainerConfig

trainer_config = GunpowderTrainerConfig(
    name="example",
    batch_size=10,
    learning_rate=0.0001,
    num_data_fetchers=8,
    snapshot_interval=100,
    min_masked=0.05,
    clip_raw=False,
)
config_store.store_trainer_config(trainer_config)

# %% [markdown]
# ## Run
# Now that we have our components configured, we just need to combine them
# into a run and start training. We can have multiple repetitions of a single
# set of configs in order to increase our chances of finding an optimum.

# %%
from dacapo.experiments import RunConfig
from dacapo.experiments.run import Run

iterations = 10000
validation_interval = iterations // 4
run_config = RunConfig(
    name="example_run",
    datasplit_config=datasplit_config,
    task_config=affs_task_config,
    architecture_config=architecture_config,
    trainer_config=trainer_config,
    num_iterations=iterations,
    validation_interval=validation_interval,
    repetition=0,
)
config_store.store_run_config(run_config)

# %% [markdown]
# ## Train
#
# NOTE: The run stats are stored in the `runs_base_dir/stats` directory.
# You can delete this directory to remove all stored stats if you want to re-run training.
# Otherwise, the stats will be appended to the existing files, and the run won't start
# from scratch. This may cause errors.

# %%
from dacapo.train import train_run
from dacapo.validate import validate
from dacapo.experiments.run import Run
from dacapo.store.create_store import create_config_store

config_store = create_config_store()

run = Run(config_store.retrieve_run_config("example_run"))
if __name__ == "__main__":
    train_run(run)

# %% [markdown]
# ## Visualize
# Let's visualize the results of the training run. DaCapo saves a few artifacts during training
# including snapshots, validation results, and the loss.

# %%
stats_store = create_stats_store()
training_stats = stats_store.retrieve_training_stats(run_config.name)
stats = training_stats.to_xarray()
plt.plot(stats)
plt.title("Training Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()

# %%
import zarr

num_snapshots = run_config.num_iterations // run_config.trainer_config.snapshot_interval
fig, ax = plt.subplots(num_snapshots, 3, figsize=(10, 2 * num_snapshots))

# Set column titles
column_titles = ["Raw", "Target", "Prediction"]
for col in range(3):
    ax[0, col].set_title(column_titles[col])

for snapshot in range(num_snapshots):
    snapshot_it = snapshot * run_config.trainer_config.snapshot_interval
    # break
    raw = zarr.open(
        f"/Users/pattonw/dacapo/example_run/snapshot.zarr/{snapshot_it}/volumes/raw"
    )[:]
    target = zarr.open(
        f"/Users/pattonw/dacapo/example_run/snapshot.zarr/{snapshot_it}/volumes/target"
    )[0]
    prediction = zarr.open(
        f"/Users/pattonw/dacapo/example_run/snapshot.zarr/{snapshot_it}/volumes/prediction"
    )[0]
    c = (raw.shape[1] - target.shape[1]) // 2
    ax[snapshot, 0].imshow(raw[raw.shape[0] // 2, c:-c, c:-c])
    ax[snapshot, 1].imshow(target[target.shape[0] // 2])
    ax[snapshot, 2].imshow(prediction[prediction.shape[0] // 2])
    ax[snapshot, 0].set_ylabel(f"Snapshot {snapshot_it}")
plt.show()
