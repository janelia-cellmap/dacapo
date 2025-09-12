import os
from ..fixtures import *

import pytest
from pytest_lazy_fixtures import lf

import logging

logging.basicConfig(level=logging.INFO)


@pytest.mark.parametrize(
    "array_config",
    [
        lf("zarr_array"),
        lf("multiscale_zarr"),
    ],
)
def test_array(array_config):
    array = array_config.array()

    assert array.offset == (
        12,
        12,
        12,
    ), f"offset is not correct, expected (12, 12, 12), got {array.offset}"
    assert array.voxel_size == (
        1,
        2,
        4,
    ), f"resolution is not correct, expected (1, 2, 4), got {array.voxel_size}"
    assert array.axis_names == [
        "z",
        "y",
        "x",
    ], f"axis names are not correct, expected ['z', 'y', 'x'], got {array.axis_names}"

    # offset = array.attrs["offset"]
    # resolution = array.attrs["resolution"]
