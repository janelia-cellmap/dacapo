from ..fixtures import *

from dacapo.store.create_store import create_config_store
from dacapo.tmp import num_channels_from_array

import pytest
from pytest_lazy_fixtures import lf
from funlib.persistence import Array


@pytest.mark.parametrize(
    "array_config",
    [
        lf("cellmap_array"),
        lf("zarr_array"),
        lf("dummy_array"),
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
    array: Array = array_config.array("r+")

    # Test API
    # channels/axis_names
    if "c^" in array.axis_names:
        assert num_channels_from_array(array) is not None
    else:
        assert num_channels_from_array(array) is None
    # dims/voxel_size/roi
    assert array.spatial_dims == array.voxel_size.dims
    assert array.spatial_dims == array.roi.dims
    # fetching data:
    expected_data_shape = array.roi.shape / array.voxel_size
    assert array[array.roi].shape[-array.spatial_dims :] == expected_data_shape
    # setting data:
    if array.is_writeable:
        data_slice = array[0]
        array[0] = data_slice + 1
        assert data_slice.sum() == 0
        assert (array[0] - data_slice).sum() == data_slice.size
