from ..fixtures import *

from dacapo.gp import DaCapoArraySource

import gunpowder as gp

import pytest
from pytest_lazy_fixtures import lf


@pytest.mark.parametrize(
    "array_config",
    [
        lf("cellmap_array"),
        lf("zarr_array"),
        lf("dummy_array"),
    ],
)
def test_gp_dacapo_array_source(array_config):
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
