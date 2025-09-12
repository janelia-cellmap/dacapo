from dacapo.experiments.datasplits.datasets.arrays.resampled_array_config import (
    ResampledArrayConfig,
)

import numpy as np
from funlib.persistence import Array
from funlib.geometry import Coordinate


def test_resample():
    # test downsampling arrays with shape 10 and 11 by a factor of 2 to test croping works
    for top in [11, 12]:
        arr = Array(np.array(np.arange(1, top)), offset=(0,), voxel_size=(3,))
        resample_config = ResampledArrayConfig(
            "test_resample", None, upsample=None, downsample=(2,), interp_order=1
        )
        resampled = resample_config.preprocess(arr)
        assert resampled.voxel_size == Coordinate((6,))
        assert resampled.shape == (5,)
        assert np.allclose(resampled[:], np.array([1.5, 3.5, 5.5, 7.5, 9.5]))

    # test 2D array
    arr = Array(
        np.array(np.arange(1, 11).reshape(5, 2).T), offset=(0, 0), voxel_size=(3, 3)
    )
    resample_config = ResampledArrayConfig(
        "test_resample", None, upsample=None, downsample=(2, 1), interp_order=1
    )
    resampled = resample_config.preprocess(arr)
    assert resampled.voxel_size == Coordinate(6, 3)
    assert resampled.shape == (1, 5)
    assert np.allclose(resampled[:], np.array([[1.5, 3.5, 5.5, 7.5, 9.5]]))
