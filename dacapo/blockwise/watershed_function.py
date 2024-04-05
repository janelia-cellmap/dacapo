import numpy as np
import numpy_indexed as npi
import mwatershed as mws
from scipy.ndimage import measurements


def segment_function(input_array, block, offsets, bias):
    """
    Segment the input array using the multicut watershed algorithm.

    Args:
        input_array (np.ndarray): The input array.
        block (daisy.Block): The block to be processed.
        offsets (List[Tuple[int]]): The offsets.
        bias (float): The bias.
    Returns:
        np.ndarray: The segmented array.
    Raises:
        NotImplementedError: If the method is not implemented in the derived class.
    Examples:
        >>> input_array = np.random.rand(128, 128, 128)
        >>> block = daisy.Block((0, 0, 0), (128, 128, 128))
        >>> offsets = [(0, 1, 0), (1, 0, 0), (0, 0, 1)]
        >>> bias = 0.1
        >>> segmentation = segment_function(input_array, block, offsets, bias)
    Note:
        The method is implemented in the class.
        If a previous segmentation is provided, it must have a "grid graph" in its metadata.
        DGA: had to add in flatten and reshape since remap (in particular indices) didn't seem to work with ndarrays for the input
    """
    # if a previous segmentation is provided, it must have a "grid graph"
    # in its metadata.
    pred_data = input_array[block.read_roi]
    affs = pred_data[: len(offsets)].astype(np.float64)
    segmentation = mws.agglom(
        affs - bias,
        offsets,
    )
    # filter fragments
    average_affs = np.mean(affs, axis=0)

    filtered_fragments = []

    fragment_ids = np.unique(segmentation)

    for fragment, mean in zip(
        fragment_ids, measurements.mean(average_affs, segmentation, fragment_ids)
    ):
        if mean < bias:
            filtered_fragments.append(fragment)

    filtered_fragments = np.array(filtered_fragments, dtype=segmentation.dtype)
    replace = np.zeros_like(filtered_fragments)

    # DGA: had to add in flatten and reshape since remap (in particular indices) didn't seem to work with ndarrays for the input
    if filtered_fragments.size > 0:
        segmentation = npi.remap(
            segmentation.flatten(), filtered_fragments, replace
        ).reshape(segmentation.shape)

    return segmentation
