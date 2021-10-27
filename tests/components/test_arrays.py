from ..fixtures.arrays import ARRAY_MK_FUNCTIONS

from dacapo import Options
from dacapo.store import create_config_store

from funlib.geometry import Coordinate, Roi

import pytest


@pytest.mark.parametrize("array_mk_function", ARRAY_MK_FUNCTIONS)
def test_arrays(tmp_path, array_mk_function):
    # Initialize the config store
    Options.instance(type="files", runs_base_dir=f"{tmp_path}")
    store = create_config_store()

    # Initialize the dataset and get the array config
    array_config = array_mk_function(tmp_path)

    # Test store/retrieve
    store.store_array_config(array_config)
    fetched_array_config = store.retrieve_array_config(array_config.name)
    assert fetched_array_config == array_config

    # Create Array from config
    array = array_config.array_type(array_config)

    # Test API
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
