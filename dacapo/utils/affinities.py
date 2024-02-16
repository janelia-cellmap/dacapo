from funlib.geometry import Coordinate
import numpy as np
import logging
from typing import List

logger = logging.getLogger(__name__)

def seg_to_affgraph(seg: np.ndarray, neighborhood: List[Coordinate]) -> np.ndarray:
    """
    Construct an affinity graph from a given segmentation image.

    Args:
        seg (np.ndarray): A segmented image for which an affinity graph is to be created.
        neighborhood (List[Coordinate]): List of neighborhood coordinates for the affinity graph.

    Returns:
        np.ndarray: An affinity graph represented as an n-dimensional array with shape (e, z, y, x) .
    """

    nhood: np.ndarray = np.array(neighborhood)
    shape = seg.shape
    nEdge = nhood.shape[0]
    dims = nhood.shape[1]
    aff = np.zeros((nEdge,) + shape, dtype=np.int32)

    if dims == 2:
        for e in range(nEdge):
            aff[
                e,
                max(0, -nhood[e, 0]) : min(shape[0], shape[0] - nhood[e, 0]),
                max(0, -nhood[e, 1]) : min(shape[1], shape[1] - nhood[e, 1]),
            ] = (
                (
                    seg[
                        max(0, -nhood[e, 0]) : min(shape[0], shape[0] - nhood[e, 0]),
                        max(0, -nhood[e, 1]) : min(shape[1], shape[1] - nhood[e, 1]),
                    ]
                    == seg[
                        max(0, nhood[e, 0]) : min(shape[0], shape[0] + nhood[e, 0]),
                        max(0, nhood[e, 1]) : min(shape[1], shape[1] + nhood[e, 1]),
                    ]
                )
                * (
                    seg[
                        max(0, -nhood[e, 0]) : min(shape[0], shape[0] - nhood[e, 0]),
                        max(0, -nhood[e, 1]) : min(shape[1], shape[1] - nhood[e, 1]),
                    ]
                    > 0
                )
                * (
                    seg[
                        max(0, nhood[e, 0]) : min(shape[0], shape[0] + nhood[e, 0]),
                        max(0, nhood[e, 1]) : min(shape[1], shape[1] + nhood[e, 1]),
                    ]
                    > 0
                )
            )

    elif dims == 3:
        for e in range(nEdge):
            aff[
                e,
                max(0, -nhood[e, 0]) : min(shape[0], shape[0] - nhood[e, 0]),
                max(0, -nhood[e, 1]) : min(shape[1], shape[1] - nhood[e, 1]),
                max(0, -nhood[e, 2]) : min(shape[2], shape[2] - nhood[e, 2]),
            ] = (
                (
                    seg[
                        max(0, -nhood[e, 0]) : min(shape[0], shape[0] - nhood[e, 0]),
                        max(0, -nhood[e, 1]) : min(shape[1], shape[1] - nhood[e, 1]),
                        max(0, -nhood[e, 2]) : min(shape[2], shape[2] - nhood[e, 2]),
                    ]
                    == seg[
                        max(0, nhood[e, 0]) : min(shape[0], shape[0] + nhood[e, 0]),
                        max(0, nhood[e, 1]) : min(shape[1], shape[1] + nhood[e, 1]),
                        max(0, nhood[e, 2]) : min(shape[2], shape[2] + nhood[e, 2]),
                    ]
                )
                * (
                    seg[
                        max(0, -nhood[e, 0]) : min(shape[0], shape[0] - nhood[e, 0]),
                        max(0, -nhood[e, 1]) : min(shape[1], shape[1] - nhood[e, 1]),
                        max(0, -nhood[e, 2]) : min(shape[2], shape[2] - nhood[e, 2]),
                    ]
                    > 0
                )
                * (
                    seg[
                        max(0, nhood[e, 0]) : min(shape[0], shape[0] + nhood[e, 0]),
                        max(0, nhood[e, 1]) : min(shape[1], shape[1] + nhood[e, 1]),
                        max(0, nhood[e, 2]) : min(shape[2], shape[2] + nhood[e, 2]),
                    ]
                    > 0
                )
            )

    else:
        raise RuntimeError(f"AddAffinities works only in 2 or 3 dimensions, not {dims}")

    return aff

def padding(neighborhood, voxel_size):
    """
    Get the appropriate padding for a given neighborhood and voxel size.

    Args:
        neighborhood: Neighborhood for which padding is to be found.
        voxel_size: Size of the voxel for which padding is to be found.

    Returns:
        Tuple: A tuple containing the negative and positive padding.
    """
    dims = voxel_size.dims
    padding_neg = (
        Coordinate(min([0] + [a[d] for a in neighborhood]) for d in range(dims))
        * voxel_size
    )

    padding_pos = (
        Coordinate(max([0] + [a[d] for a in neighborhood]) for d in range(dims))
        * voxel_size
    )
    return padding_neg, padding_pos