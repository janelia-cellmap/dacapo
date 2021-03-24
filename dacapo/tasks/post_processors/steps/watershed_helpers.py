import daisy
import numpy as np
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage import measurements, label
from funlib.segment.arrays import relabel
from funlib.segment.arrays import replace_values
from skimage.segmentation import watershed
import waterz

import logging

logger = logging.getLogger(__name__)


def watershed_in_block(
    affs,
    block,
    context,
    rag_provider,
    fragments_out,
    num_voxels_in_block,
    fragments_in_xy,
    epsilon_agglomerate,
    filter_fragments,
    min_seed_distance,
    mask=None,
):
    """
    Args:
        filter_fragments (float):
            Filter fragments that have an average affinity lower than this
            value.
        min_seed_distance (int):
            Controls distance between seeds in the initial watershed. Reducing
            this value improves downsampled segmentation.
        epsilon_agglomerate (float):
            Automatically agglomerate fragments with extremely high affinity to
            reduce the total number of fragmets
        min_seed_distance (float):
            The minimum distance between two watershed seeds
    """

    logger.debug("reading affs from %s", block.read_roi)

    affs = affs.intersect(block.read_roi)
    affs.materialize()

    if affs.dtype == np.uint8:
        logger.info("Assuming affinities are in [0,255]")
        max_affinity_value = 255.0
        affs.data = affs.data.astype(np.float32)
    else:
        max_affinity_value = 1.0

    if mask is not None:

        logger.debug("reading mask from %s", block.read_roi)
        mask_data = get_mask_data_in_roi(mask, affs.roi, affs.voxel_size)
        logger.debug("masking affinities")
        affs.data *= mask_data


    # extract fragments
    fragments_data = watershed_from_affinities(
        affs.data,
        max_affinity_value,
        fragments_in_xy=False,
        min_seed_distance=min_seed_distance,
        voxel_size=affs.voxel_size,
    )

    if mask is not None:
        fragments_data *= mask_data.astype(np.uint64)

    if filter_fragments > 0:

        if fragments_in_xy:
            average_affs = np.mean(affs.data[0:2] / max_affinity_value, axis=0)
        else:
            average_affs = np.mean(affs.data / max_affinity_value, axis=0)

        filtered_fragments = []

        fragment_ids = np.unique(fragments_data)

        for fragment, mean in zip(
            fragment_ids, measurements.mean(average_affs, fragments_data, fragment_ids)
        ):
            if mean < filter_fragments:
                filtered_fragments.append(fragment)

        filtered_fragments = np.array(filtered_fragments, dtype=fragments_data.dtype)
        replace = np.zeros_like(filtered_fragments)
        replace_values(fragments_data, filtered_fragments, replace, inplace=True)

    if epsilon_agglomerate > 0:

        logger.info(
            "Performing initial fragment agglomeration until %f", epsilon_agglomerate
        )

        generator = waterz.agglomerate(
            affs=affs.data / max_affinity_value,
            thresholds=[epsilon_agglomerate],
            fragments=fragments_data,
            scoring_function="OneMinus<HistogramQuantileAffinity<RegionGraphType, 25, ScoreValue, 256, false>>",
            discretize_queue=256,
            return_merge_history=False,
            return_region_graph=False,
        )
        fragments_data[:] = next(generator)

        # cleanup generator
        for _ in generator:
            pass

    # TODO: add key value replacement option

    fragments = daisy.Array(fragments_data, affs.roi, affs.voxel_size)

    # crop fragments to write_roi
    fragments = fragments[block.write_roi]
    fragments.materialize()
    max_id = fragments.data.max()

    # ensure we don't have IDs larger than the number of voxels (that would
    # break uniqueness of IDs below)
    if max_id > num_voxels_in_block:
        logger.debug(
            "fragments in %s have max ID %d, relabelling...", block.write_roi, max_id
        )
        fragments.data, max_id = relabel(fragments.data)

        assert max_id < num_voxels_in_block

    # ensure unique IDs
    id_bump = block.block_id * num_voxels_in_block
    logger.debug(
        f"bumping fragment IDs by {block.block_id} * {num_voxels_in_block} = {id_bump}"
    )
    fragments.data[fragments.data > 0] += id_bump
    fragment_ids = range(id_bump + 1, id_bump + 1 + int(max_id))

    # store fragments
    logger.debug("writing fragments to %s", block.write_roi)
    fragments_out[block.write_roi] = fragments

    # following only makes a difference if fragments were found
    if max_id == 0:
        return

    # get fragment centers
    fragment_centers = {
        fragment: block.write_roi.get_offset() + affs.voxel_size * center
        for fragment, center in zip(
            fragment_ids,
            measurements.center_of_mass(fragments.data, fragments.data, fragment_ids),
        )
        if not np.isnan(center[0])
    }

    # store nodes
    rag = rag_provider[block.write_roi]
    rag.add_nodes_from(
        [
            (node, {"center_z": c[0], "center_y": c[1], "center_x": c[2]})
            for node, c in fragment_centers.items()
        ]
    )
    rag.write_nodes(block.write_roi)


def watershed_from_affinities(
    affs,
    max_affinity_value=1.0,
    fragments_in_xy=False,
    return_seeds=False,
    min_seed_distance=1,
    voxel_size=1,
    compactness=0,
):
    """Extract initial fragments from affinities using a watershed
    transform. Returns the fragments and the maximal ID in it.
    Returns:
        (fragments, max_id)
        or
        (fragments, max_id, seeds) if return_seeds == True"""

    boundary_mask = np.mean(affs, axis=0) > 0.5 * max_affinity_value
    boundary_distances = distance_transform_edt(boundary_mask)

    neighborhood_size = 1 + 2 * np.ceil(min_seed_distance / np.array(voxel_size))
    max_filtered = maximum_filter(boundary_distances, neighborhood_size)
    maxima = max_filtered == boundary_distances
    seeds, n = label(maxima)

    fragments = watershed(
        boundary_distances.max() - boundary_distances,
        markers=seeds,
        connectivity=1,
        offset=None,
        mask=boundary_mask,
        compactness=compactness,
        watershed_line=False,
    ).astype(np.uint64)

    return fragments
