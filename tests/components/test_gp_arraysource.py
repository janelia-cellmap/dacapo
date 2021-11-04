from ..fixtures.arrays import ARRAY_MK_FUNCTIONS

from dacapo.gp import DaCapoArraySource

import gunpowder as gp

import pytest


@pytest.mark.parametrize("array_mk_function", ARRAY_MK_FUNCTIONS)
def test_gp_dacapo_array_source(tmp_path, array_mk_function):

    # Initialize the dataset and get the array config
    array_config = array_mk_function(tmp_path)

    # Create Array from config
    array = array_config.array_type(array_config)

    # Make sure the DaCapoArraySource can properly read
    # the data in `array`
    key = gp.ArrayKey("TEST")
    source_node = DaCapoArraySource(array, key)

    with gp.build(source_node):
        request = gp.BatchRequest()
        request[key] = gp.ArraySpec(roi=array.roi)
        batch = source_node.request_batch(request)
        data = batch[key].data
        assert (data - array[array.roi]).sum() == 0
