from funlib.geometry import Coordinate

import numpy as np

import logging
from typing import List

logger = logging.getLogger(__name__)


def seg_to_affgraph(seg: np.ndarray, neighborhood: List[Coordinate]) -> np.ndarray:
    nhood: np.ndarray = np.array(neighborhood)

    # constructs an affinity graph from a segmentation
    # assume affinity graph is represented as:
    # shape = (e, z, y, x)
    # nhood.shape = (edges, 3)
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
    Get the appropriate padding to make sure all provided affinities are "True"
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
