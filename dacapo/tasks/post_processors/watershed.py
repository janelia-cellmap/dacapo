import daisy
import numpy as np

from .post_processor import PostProcessor

from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import distance_transform_edt
import mahotas
import waterz
import lsd

from dacapo.store import MongoDbStore

import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class Watershed(PostProcessor):
    def set_prediction(self, prediction):
        fragments = watershed_from_affinities(prediction.to_ndarray())[0]
        thresholds = self.parameter_range.threshold
        segmentations = waterz.agglomerate(
            affs=prediction.to_ndarray(),
            fragments=fragments,
            thresholds=thresholds,
        )

        self.segmentations = {t: s for t, s in zip(thresholds, segmentations)}

    def process(self, prediction, parameters):
        seg = self.segmentations[parameters.threshold]
        return daisy.Array(
            seg,
            roi=prediction.roi,
            voxel_size=prediction.voxel_size,
        )

    def daisy_steps(self):
        yield ("fragments", "shrink", blockwise_fragments_worker)
        yield ("agglomerate", "shrink", blockwise_agglomerate_worker)


def blockwise_fragments_worker(
    run_hash,
    affs_file,
    affs_dataset,
    fragments_file,
    fragments_dataset,
    mask_file=None,
    mask_dataset=None,
    filter_fragments=0,
    fragments_in_xy=True,
    epsilon_agglomerate=0,
):

    logger.info("Reading affs from %s", affs_file)
    affs = daisy.open_ds(affs_file, affs_dataset, mode="r")

    logger.info("Reading fragments from %s", fragments_file)
    if not Path(fragments_file, fragments_dataset).exists():
        daisy.prepare_ds(
            fragments_file,
            fragments_dataset,
            affs.roi,
            affs.voxel_size,
            dtype=np.uint64,
            write_size=affs.voxel_size * affs.chunk_shape[-len(affs.voxel_size) :],
        )
    fragments = daisy.open_ds(fragments_file, fragments_dataset, mode="r+")

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
    affs_file,
    affs_dataset,
    fragments_file,
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

    logging.info("Reading affs from %s" % affs_file)
    affs = daisy.open_ds(affs_file, affs_dataset, mode="r")
    fragments = daisy.open_ds(fragments_file, fragments_dataset, mode="r+")

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


def watershed_from_affinities(
    affs,
    max_affinity_value=1.0,
    fragments_in_xy=False,
    return_seeds=False,
    min_seed_distance=10,
):
    """Extract initial fragments from affinities using a watershed
    transform. Returns the fragments and the maximal ID in it.
    Returns:
        (fragments, max_id)
        or
        (fragments, max_id, seeds) if return_seeds == True"""

    if fragments_in_xy:

        mean_affs = 0.5 * (affs[1] + affs[2])
        depth = mean_affs.shape[0]

        fragments = np.zeros(mean_affs.shape, dtype=np.uint64)
        if return_seeds:
            seeds = np.zeros(mean_affs.shape, dtype=np.uint64)

        id_offset = 0
        for z in range(depth):

            boundary_mask = mean_affs[z] > 0.5 * max_affinity_value
            boundary_distances = distance_transform_edt(boundary_mask)

            ret = watershed_from_boundary_distance(
                boundary_distances,
                return_seeds=return_seeds,
                id_offset=id_offset,
                min_seed_distance=min_seed_distance,
            )

            fragments[z] = ret[0]
            if return_seeds:
                seeds[z] = ret[2]

            id_offset = ret[1]

        ret = (fragments, id_offset)
        if return_seeds:
            ret += (seeds,)

    else:

        boundary_mask = np.mean(affs, axis=0) > 0.5 * max_affinity_value
        boundary_distances = distance_transform_edt(boundary_mask)

        ret = watershed_from_boundary_distance(
            boundary_distances, return_seeds, min_seed_distance=min_seed_distance
        )

        fragments = ret[0]

    return ret


def watershed_from_boundary_distance(
    boundary_distances, return_seeds=False, id_offset=0, min_seed_distance=10
):

    max_filtered = maximum_filter(boundary_distances, min_seed_distance)
    maxima = max_filtered == boundary_distances
    seeds, n = mahotas.label(maxima)

    if n == 0:
        return np.zeros(boundary_distances.shape, dtype=np.uint64), id_offset

    seeds[seeds != 0] += id_offset

    fragments = mahotas.cwatershed(boundary_distances.max() - boundary_distances, seeds)

    ret = (fragments.astype(np.uint64), n + id_offset)
    if return_seeds:
        ret = ret + (seeds.astype(np.uint64),)

    return ret