from .dummy_array import mk_dummy_array
from .zarr_array import mk_zarr_array
from .cellmap_array import mk_cellmap_array

ARRAY_MK_FUNCTIONS = [mk_dummy_array, mk_zarr_array, mk_cellmap_array]
