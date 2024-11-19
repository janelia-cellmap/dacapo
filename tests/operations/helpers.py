import numpy as np
from funlib.persistence import prepare_ds
from funlib.geometry import Coordinate

from dacapo.experiments.datasplits import SimpleDataSplitConfig
from dacapo.experiments.tasks import (
    DistanceTaskConfig,
    OneHotTaskConfig,
    AffinitiesTaskConfig,
)
from dacapo.experiments.architectures import CNNectomeUNetConfig

from pathlib import Path


def build_test_data_config(
    tmpdir: Path, data_dims: int, channels: bool, upsample: bool, task_type: str
):
    """
    Builds the simplest possible datasplit given the parameters.

    Labels are alternating planes/lines of 0/1 in the last dimension.
    Intensities are random where labels are > 0, else 0. (If channels, stack twice.)
    if task_type is "semantic", labels are binarized via labels > 0.

    if upsampling, labels are upsampled by a factor of 2 in each dimension
    """

    data_shape = (64, 64, 64)[-data_dims:]
    mesh = np.meshgrid(
        *[np.linspace(0, dim - 1, dim * (1 + upsample)) for dim in data_shape]
    )
    labels = mesh[-1] * (mesh[-1] % 2 > 0.75)

    intensities = np.random.rand(*labels.shape) * labels > 0

    if channels:
        intensities = np.stack([intensities, intensities], axis=0)

    intensities_array = prepare_ds(
        tmpdir / "test_data.zarr/raw",
        intensities.shape,
        offset=(0,) * data_dims,
        voxel_size=(2,) * data_dims,
        dtype=intensities.dtype,
        mode="w",
    )
    intensities_array[:] = intensities

    if task_type == "semantic":
        labels = labels > 0

    labels_array = prepare_ds(
        tmpdir / "test_data.zarr/labels",
        labels.shape,
        offset=(0,) * data_dims,
        voxel_size=(2 - upsample,) * data_dims,
        dtype=labels.dtype,
        mode="w",
    )
    labels_array[:] = labels

    return SimpleDataSplitConfig(name="test_data", path=tmpdir / "test_data.zarr")


def build_test_task_config(task, data_dims: int, architecture_dims: int):
    """
    Build the simplest task config given the parameters.
    """
    if task == "distance":
        return DistanceTaskConfig(
            name="test_distance_task",
            channels=["fg"],
            clip_distance=4,
            tol_distance=4,
            scale_factor=8,
        )
    if task == "onehot":
        return OneHotTaskConfig(
            name="test_onehot_task", classes=["bg", "fg"], kernel_size=1
        )
    if task == "affs":
        # TODO: should configs be able to take any sequence for the neighborhood?
        if data_dims == 2:
            # 2D
            neighborhood = [Coordinate(1, 0), Coordinate(0, 1)]
        elif data_dims == 3 and architecture_dims == 2:
            # 3D but only generate 2D affs
            neighborhood = [Coordinate(0, 1, 0), Coordinate(0, 0, 1)]
        elif data_dims == 3 and architecture_dims == 3:
            # 3D
            neighborhood = [
                Coordinate(1, 0, 0),
                Coordinate(0, 1, 0),
                Coordinate(0, 0, 1),
            ]
        return AffinitiesTaskConfig(name="test_affs_task", neighborhood=neighborhood)


def build_test_architecture_config(
    data_dims: int,
    architecture_dims: int,
    channels: bool,
    batch_norm: bool,
    upsample: bool,
    use_attention: bool,
    padding: str,
):
    """
    Build the simplest architecture config given the parameters.
    """
    if data_dims == 2:
        input_shape = (18, 18)
        downsample_factors = [(2, 2)]
        upsample_factors = [(2, 2)] * int(upsample)

        kernel_size_down = [[(3, 3)] * 2] * 2
        kernel_size_up = [[(3, 3)] * 2] * 1
        kernel_size_down = None  # the default should work
        kernel_size_up = None  # the default should work

    elif data_dims == 3 and architecture_dims == 2:
        input_shape = (1, 18, 18)
        downsample_factors = [(1, 2, 2)]

        # test data upsamples in all dimensions so we have
        # to here too
        upsample_factors = [(2, 2, 2)] * int(upsample)

        # we have to force the 3D kernels to be 2D
        kernel_size_down = [[(1, 3, 3)] * 2] * 2
        kernel_size_up = [[(1, 3, 3)] * 2] * 1

    elif data_dims == 3 and architecture_dims == 3:
        input_shape = (18, 18, 18)
        downsample_factors = [(2, 2, 2)]
        upsample_factors = [(2, 2, 2)] * int(upsample)

        kernel_size_down = [[(3, 3, 3)] * 2] * 2
        kernel_size_up = [[(3, 3, 3)] * 2] * 1
        kernel_size_down = None  # the default should work
        kernel_size_up = None  # the default should work

    return CNNectomeUNetConfig(
        name="test_cnnectome_unet",
        input_shape=input_shape,
        eval_shape_increase=input_shape,
        fmaps_in=1 + channels,
        num_fmaps=2,
        fmaps_out=2,
        fmap_inc_factor=2,
        downsample_factors=downsample_factors,
        kernel_size_down=kernel_size_down,
        kernel_size_up=kernel_size_up,
        constant_upsample=True,
        upsample_factors=upsample_factors,
        batch_norm=batch_norm,
        use_attention=use_attention,
        padding=padding,
    )
