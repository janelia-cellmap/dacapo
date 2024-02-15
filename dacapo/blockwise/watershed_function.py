import numpy as np
import numpy_indexed as npi
import mwatershed as mws
from scipy.ndimage import measurements


def segment_function(input_array, block, parameters):
    # if a previous segmentation is provided, it must have a "grid graph"
    # in its metadata.
    pred_data = input_array[block.read_roi]
    affs = pred_data[: len(parameters["offsets"])].astype(np.float64)
    segmentation = mws.agglom(
        affs - parameters["bias"],
        parameters["offsets"],  # type: ignore
    )
    # filter fragments
    average_affs = np.mean(affs, axis=0)

    filtered_fragments = []

    fragment_ids = np.unique(segmentation)

    for fragment, mean in zip(
        fragment_ids, measurements.mean(average_affs, segmentation, fragment_ids)
    ):
        if mean < parameters["bias"]:
            filtered_fragments.append(fragment)

    filtered_fragments = np.array(filtered_fragments, dtype=segmentation.dtype)
    replace = np.zeros_like(filtered_fragments)

    # DGA: had to add in flatten and reshape since remap (in particular indices) didn't seem to work with ndarrays for the input
    if filtered_fragments.size > 0:
        segmentation = npi.remap(
            segmentation.flatten(), filtered_fragments, replace
        ).reshape(segmentation.shape)

    return segmentation
