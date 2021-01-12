import daisy
import numpy as np

from .post_processor import PostProcessor

from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage import label
from skimage.segmentation import watershed
import waterz
import lsd

from funlib.segment.graphs.impl import connected_components
from funlib.segment.arrays import replace_values

from dacapo.store import MongoDbStore

import logging
import time

logger = logging.getLogger(__name__)


class Watershed(PostProcessor):
    def set_prediction(self, prediction):
        fragments = watershed_from_affinities(
            prediction.to_ndarray(), prediction.voxel_size
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
        yield ("segment", "shrink", blockwise_write_segmentation, ["volumes/segmentation"])


def blockwise_fragments_worker(
    run_hash,
    output_dir,
    output_filename,
    affs_dataset,
    fragments_dataset,
    mask_file=None,
    mask_dataset=None,
    filter_fragments=0,
    fragments_in_xy=True,
    epsilon_agglomerate=0,
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

        num_voxels_in_block = (block.write_roi / affs.voxel_size).size()

        start = time.time()

        logger.info("block read roi begin: %s", block.read_roi.get_begin())
        logger.info("block read roi shape: %s", block.read_roi.get_shape())
        logger.info("block write roi begin: %s", block.write_roi.get_begin())
        logger.info("block write roi shape: %s", block.write_roi.get_shape())

        context = (block.read_roi.get_shape() - block.write_roi.get_shape()) / 2

        lsd.watershed_in_block(
            affs,
            block,
            context,
            rag_provider,
            fragments,
            num_voxels_in_block=num_voxels_in_block,
            mask=mask,
            fragments_in_xy=fragments_in_xy,
            epsilon_agglomerate=epsilon_agglomerate,
            filter_fragments=filter_fragments,
            replace_sections=None,
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
    output_dir,
    output_filename,
    fragments_dataset,
    num_workers,
    threshold,
    lookup,
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

    fragments = daisy.open_ds(f"{output_dir}/{output_filename}", fragments_dataset)
    roi = fragments.roi

    g = graph_provider.get_graph(roi)

    logger.info("Read graph in %.3fs" % (time.time() - start))

    assert g.number_of_nodes() > 0, f"No nodes found in roi {roi}"

    nodes = np.array(list(g.nodes()))
    edges = np.array([(u, v) for u, v in g.edges()], dtype=np.uint64)
    scores = np.array([attrs["merge_score"] for edge, attrs in g.edges.items()])

    logger.debug("Nodes dtype: ", nodes.dtype)
    logger.debug("edges dtype: ", edges.dtype)
    logger.debug("scores dtype: ", scores.dtype)

    logger.info("Complete RAG contains %d nodes, %d edges" % (len(nodes), len(edges)))

    start = time.time()

    logger.info("Getting CCs for threshold %.3f..." % threshold)
    start = time.time()
    components = connected_components(nodes, edges, scores, threshold)
    logger.info("%.3fs" % (time.time() - start))

    logger.info("Creating fragment-segment LUT for threshold %.3f..." % threshold)
    start = time.time()
    lut = np.array([nodes, components])

    logger.info("%.3fs" % (time.time() - start))

    logger.info("Storing fragment-segment LUT for threshold %.3f..." % threshold)
    start = time.time()

    for node, component in lut:
        g.nodes[node][lookup] = component

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
        f"{output_dir}/{output_filename}", segmentation_dataset
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

    pred_id = f"{run_hash}_segmentation"
    step_id = "prediction"

    client = daisy.Client()
    while True:

        block = client.acquire_block()
        fragments = fragments.to_ndarray(block.write_roi)
        logging.info("%.3fs" % (time.time() - start))

        start = time.time()

        relabelled = np.zeros_like(fragments)

        agg_graph = graph_provider.get_graph(roi=block.write_roi)

        lut = np.array([(node, attrs[lookup]) for node, attrs in agg_graph.nodes.items()])

        relabelled = replace_values(fragments, lut[0], lut[1], out_array=relabelled)
        logging.info("%.3fs" % (time.time() - start))

        segmentation[block.write_roi] = relabelled

        store.mark_block_done(
            pred_id, step_id, block.block_id, start, time.time() - start
        )

        client.release_block(block, ret=0)


def watershed_from_affinities(
    affs,
    voxel_size,
    max_affinity_value=1.0,
    fragments_in_xy=False,
    return_seeds=False,
    min_seed_distance=1,
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

    max_filtered = maximum_filter(
        boundary_distances, np.ceil(min_seed_distance / np.array(voxel_size))
    )
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
