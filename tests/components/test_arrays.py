from ..fixtures import *

from dacapo.store import create_config_store

import pytest
from pytest_lazyfixture import lazy_fixture


@pytest.mark.parametrize(
    "array_config",
    [
        lazy_fixture("cellmap_array"),
        lazy_fixture("zarr_array"),
        lazy_fixture("dummy_array"),
    ],
)
def test_array_api(options, array_config):
    # create_config_store (uses options behind the scenes)
    store = create_config_store()

    # Test store/retrieve
    store.store_array_config(array_config)
    fetched_array_config = store.retrieve_array_config(array_config.name)
    assert fetched_array_config == array_config

    # Create Array from config
    array = array_config.array_type(array_config)

    # Test API
    # channels/axes
    if "c" in array.axes:
        assert array.num_channels is not None
    else:
        assert array.num_channels is None
    # dims/voxel_size/roi
    assert array.dims == array.voxel_size.dims
    assert array.dims == array.roi.dims
    # fetching data:
    expected_data_shape = array.roi.shape / array.voxel_size
    assert array[array.roi].shape[-array.dims :] == expected_data_shape
    # setting data:
    if array.writable:
        data_slice = array.data[0].copy()
        array.data[0] = data_slice + 1
        assert data_slice.sum() == 0
        assert (array.data[0] - data_slice).sum() == data_slice.size
