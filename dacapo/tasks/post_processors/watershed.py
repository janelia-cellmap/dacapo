import daisy
import numpy as np

from .post_processor_abc import PostProcessorABC

from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage import measurements, label
from funlib.segment.arrays import relabel
from skimage.segmentation import watershed
import waterz
import lsd

from funlib.segment.graphs.impl import connected_components
from funlib.segment.arrays import replace_values

from dacapo.store import MongoDbStore

import logging
import time

logger = logging.getLogger(__name__)


class Watershed(PostProcessorABC):
    def set_prediction(self, prediction):
        fragments = watershed_from_affinities(
            prediction.to_ndarray(), voxel_size=prediction.voxel_size
        )
        thresholds = self.parameter_range.threshold
        segmentations = waterz.agglomerate(
            affs=prediction.to_ndarray(),
            fragments=fragments,
            thresholds=thresholds,
        )

        self.segmentations = {t: s.copy() for t, s in zip(thresholds, segmentations)}

    def process(self, prediction, parameters):
        seg = self.segmentations[parameters.threshold]
        return daisy.Array(
            seg,
            roi=prediction.roi,
            voxel_size=prediction.voxel_size,
        )

    def daisy_steps(self):
        yield ("fragments", "shrink", blockwise_fragments_worker, ["volumes/fragments"])
        yield ("agglomerate", "shrink", blockwise_agglomerate_worker, [])
        yield ("create_luts", "global", global_create_luts, [])
        yield (
            "segment",
            "shrink",
            blockwise_write_segmentation,
            ["volumes/segmentation"],
        )


def blockwise_fragments_worker(
    run_hash,
    output_dir,
    output_filename,
    affs_dataset,
    fragments_dataset,
    mask_file=None,
    mask_dataset=None,
    filter_fragments=0,
    fragments_in_xy=False,
    epsilon_agglomerate=0,
    num_voxels_in_block=1e6,
):

    logger.info(f"Reading affs from {output_dir}/{output_filename}")
    affs = daisy.open_ds(f"{output_dir}/{output_filename}", affs_dataset, mode="r")

    logger.info(f"Reading fragments from {output_dir}/{output_filename}")
    fragments = daisy.open_ds(
        f"{output_dir}/{output_filename}", fragments_dataset, mode="r+"
    )

    if mask_file:

        logger.info("Reading mask from %s", mask_file)
        mask = daisy.open_ds(mask_file, mask_dataset, mode="r")

    else:

        mask = None

    # open block done DB
    logger.warning("Mongo storage should be handled by dacapo.Store class")

    pred_id = f"{run_hash}_fragments"
    step_id = "prediction"
    store = MongoDbStore()

    # open RAG DB
    logger.info("Opening RAG DB...")
    logger.warning("Mongo storage should be handled by dacapo.Store class")
    logger.warning("Hard coded position attrs")
    rag_provider = daisy.persistence.MongoDbGraphProvider(
        store.db_name,
        host=store.db_host,
        nodes_collection=f"{run_hash}_frags",
        edges_collection=f"{run_hash}_frag_agglom",
        mode="r+",
        directed=False,
        position_attribute=["center_z", "center_y", "center_x"],
    )
    logger.info("RAG DB opened")

    client = daisy.Client()

    while True:

        block = client.acquire_block()

        if block is None:
            break

        start = time.time()

        logger.info("block read roi begin: %s", block.read_roi.get_begin())
        logger.info("block read roi shape: %s", block.read_roi.get_shape())
        logger.info("block write roi begin: %s", block.write_roi.get_begin())
        logger.info("block write roi shape: %s", block.write_roi.get_shape())

        context = (block.read_roi.get_shape() - block.write_roi.get_shape()) / 2

        watershed_in_block(
            affs,
            block,
            context,
            rag_provider,
            fragments,
            num_voxels_in_block=int(num_voxels_in_block),
            mask=mask,
            fragments_in_xy=fragments_in_xy,
            epsilon_agglomerate=epsilon_agglomerate,
            filter_fragments=filter_fragments,
        )

        store.mark_block_done(
            pred_id, step_id, block.block_id, start, time.time() - start
        )

        client.release_block(block, ret=0)


def blockwise_agglomerate_worker(
    run_hash,
    output_dir,
    output_filename,
    affs_dataset,
    fragments_dataset,
    merge_function="mean",
):

    waterz_merge_function = {
        "hist_quant_10": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 10, ScoreValue, 256, false>>",
        "hist_quant_10_initmax": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 10, ScoreValue, 256, true>>",
        "hist_quant_25": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 25, ScoreValue, 256, false>>",
        "hist_quant_25_initmax": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 25, ScoreValue, 256, true>>",
        "hist_quant_50": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256, false>>",
        "hist_quant_50_initmax": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256, true>>",
        "hist_quant_75": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 75, ScoreValue, 256, false>>",
        "hist_quant_75_initmax": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 75, ScoreValue, 256, true>>",
        "hist_quant_90": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 90, ScoreValue, 256, false>>",
        "hist_quant_90_initmax": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 90, ScoreValue, 256, true>>",
        "mean": "OneMinus<MeanAffinity<RegionGraphType, ScoreValue>>",
    }[
        merge_function
    ]  # watershed should know this

    logging.info(f"Reading affs from {output_dir}/{output_filename}")
    affs = daisy.open_ds(f"{output_dir}/{output_filename}", affs_dataset, mode="r")
    fragments = daisy.open_ds(
        f"{output_dir}/{output_filename}", fragments_dataset, mode="r+"
    )

    # open block done DB
    logger.warning("Dacapo.Store should handle mongodb storage")

    pred_id = f"{run_hash}_agglomerate"
    step_id = "prediction"
    store = MongoDbStore()

    # open RAG DB
    logging.info("Opening RAG DB...")
    logger.warning("Dacapo.Store should handle mongodb storage")
    logger.warning("Hard coded position attrs")
    rag_provider = daisy.persistence.MongoDbGraphProvider(
        store.db_name,
        host=store.db_host,
        mode="r+",
        directed=False,
        nodes_collection=f"{run_hash}_frags",
        edges_collection=f"{run_hash}_frag_agglom",
        position_attribute=["center_z", "center_y", "center_x"],
    )
    logging.info("RAG DB opened")

    client = daisy.Client()
    while True:

        block = client.acquire_block()

        if block is None:
            break

        start = time.time()

        lsd.agglomerate_in_block(
            affs,
            fragments,
            rag_provider,
            block,
            merge_function=waterz_merge_function,
            threshold=1.0,
        )

        store.mark_block_done(
            pred_id, step_id, block.block_id, start, time.time() - start
        )

        client.release_block(block, ret=0)


def global_create_luts(
    run_hash,
    threshold,
    lookup,
    roi,
):

    """

    Args:

        fragments_file (``string``):

            Path to the file containing the fragments.

        edges_collection (``string``):

            The name of the MongoDB database collection to use.

        thresholds_minmax (``list`` of ``int``):

            The lower and upper bound to use (i.e [0,1]) when generating
            thresholds.

        thresholds_step (``float``):

            The step size to use when generating thresholds between min/max.

        roi_offset (array-like of ``int``):

            The starting point (inclusive) of the ROI. Entries can be ``None``
            to indicate unboundedness.

        roi_shape (array-like of ``int``):

            The shape of the ROI. Entries can be ``None`` to indicate
            unboundedness.

    """
    store = MongoDbStore()

    start = time.time()

    edges_collection = f"{run_hash}_frag_agglom"
    graph_provider = daisy.persistence.MongoDbGraphProvider(
        store.db_name,
        store.db_host,
        nodes_collection=f"{run_hash}_frags",
        edges_collection=edges_collection,
        position_attribute=["center_z", "center_y", "center_x"],
    )

    g = graph_provider.get_graph(roi)

    logger.info("Read graph in %.3fs" % (time.time() - start))

    assert g.number_of_nodes() > 0, f"No nodes found in roi {roi}"

    nodes = np.array(list(g.nodes()))
    edges = np.array([(u, v) for u, v in g.edges()], dtype=np.uint64)
    scores = np.array(
        [attrs["merge_score"] for edge, attrs in g.edges.items()], dtype=np.float32
    )
    scores = np.nan_to_num(scores, nan=1)

    logger.debug(
        f"percentiles (1, 5, 50, 95, 99): {np.percentile(scores, [1,5,50,95,99])}"
    )

    logger.debug("Nodes dtype: ", nodes.dtype)
    logger.debug("edges dtype: ", edges.dtype)
    logger.debug("scores dtype: ", scores.dtype)

    logger.debug("Complete RAG contains %d nodes, %d edges" % (len(nodes), len(edges)))

    start = time.time()

    logger.debug("Getting CCs for threshold %.3f..." % threshold)
    start = time.time()
    components = connected_components(nodes, edges, scores, threshold).astype(np.int64)
    logger.debug("%.3fs" % (time.time() - start))

    logger.debug("Creating fragment-segment LUT for threshold %.3f..." % threshold)
    start = time.time()
    lut = np.array([(n, c) for n, c in zip(nodes, components)], dtype=int)

    logger.info("%.3fs" % (time.time() - start))

    logger.info("Storing fragment-segment LUT for threshold %.3f..." % threshold)
    start = time.time()

    for node, component in lut:
        g.nodes[node][lookup] = int(component)

    g.update_node_attrs(attributes=[lookup])

    logger.info("%.3fs" % (time.time() - start))

    logger.info("Created and stored lookup tables in %.3fs" % (time.time() - start))

    return True


def blockwise_write_segmentation(
    run_hash,
    output_dir,
    output_filename,
    fragments_dataset,
    segmentation_dataset,
    lookup,
):

    logging.info("Copying fragments to memory...")
    start = time.time()
    fragments = daisy.open_ds(f"{output_dir}/{output_filename}", fragments_dataset)
    segmentation = daisy.open_ds(
        f"{output_dir}/{output_filename}", segmentation_dataset, mode="r+"
    )

    store = MongoDbStore()

    edges_collection = f"{run_hash}_frag_agglom"
    graph_provider = daisy.persistence.MongoDbGraphProvider(
        store.db_name,
        store.db_host,
        nodes_collection=f"{run_hash}_frags",
        edges_collection=edges_collection,
        position_attribute=["center_z", "center_y", "center_x"],
    )

    pred_id = f"{run_hash}_segment"
    step_id = "prediction"

    client = daisy.Client()
    while True:

        block = client.acquire_block()
        if block is None:
            break

        block_fragments = fragments.to_ndarray(block.write_roi)
        logging.info("%.3fs" % (time.time() - start))

        start = time.time()

        relabelled = np.zeros_like(block_fragments)

        agg_graph = graph_provider.get_graph(roi=block.write_roi)

        lut = np.array(
            [
                (np.uint64(node), np.uint64(attrs[lookup]))
                for node, attrs in agg_graph.nodes.items()
                if lookup in attrs
            ]
        )

        replace_values(block_fragments, lut[:, 0], lut[:, 1], out_array=relabelled)
        logging.info("%.3fs" % (time.time() - start))

        segmentation[block.write_roi] = relabelled

        store.mark_block_done(
            pred_id, step_id, block.block_id, start, time.time() - start
        )

        client.release_block(block, ret=0)


def watershed_in_block(
    affs,
    block,
    context,
    rag_provider,
    fragments_out,
    num_voxels_in_block,
    mask=None,
    fragments_in_xy=False,
    epsilon_agglomerate=0.0,
    filter_fragments=0.0,
    min_seed_distance=10,
):
    """
    Args:
        filter_fragments (float):
            Filter fragments that have an average affinity lower than this
            value.
        min_seed_distance (int):
            Controls distance between seeds in the initial watershed. Reducing
            this value improves downsampled segmentation.
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

    """
    if mask is not None:

        logger.debug("reading mask from %s", block.read_roi)
        mask_data = get_mask_data_in_roi(mask, affs.roi, affs.voxel_size)
        logger.debug("masking affinities")
        affs.data *= mask_data
    """

    # extract fragments
    fragments_data = watershed_from_affinities(
        affs.data,
        max_affinity_value,
        fragments_in_xy=False,
        min_seed_distance=min_seed_distance,
        voxel_size=affs.voxel_size,
    )

    """
    if mask is not None:
        fragments_data *= mask_data.astype(np.uint64)
    """

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

    # todo add key value replacement option

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
