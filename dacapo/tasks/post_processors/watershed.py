import daisy
import numpy as np

from .post_processor import PostProcessor

from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import distance_transform_edt
import mahotas
import waterz
import lsd

import logging
import time

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

    def blockwise_fragments_worker(self, config_file):
        """
        daisy call:

            daisy.run_blockwise(
                total_roi=total_roi,
                read_roi=read_roi,
                write_roi=write_roi,
                process_function=lambda: start_worker(
                    affs_file,
                    affs_dataset,
                    fragments_file,
                    fragments_dataset,
                    db_host,
                    db_name,
                    context,
                    fragments_in_xy,
                    queue,
                    network_dir,
                    epsilon_agglomerate,
                    mask_file,
                    mask_dataset,
                    filter_fragments,
                    replace_sections,
                    num_voxels_in_block),
                check_function=lambda b: check_block(
                    blocks_extracted,
                    b),
                num_workers=num_workers,
                read_write_conflict=False,
                fit='shrink')
        """
        config = None

        logger.info(config)

        affs_file = config["affs_file"]  # dacapo knows
        affs_dataset = config["affs_dataset"]  # dacapo knows
        fragments_file = config["fragments_file"]  # dacapo knows
        fragments_dataset = config["fragments_dataset"]  # dacapo knows
        mask_file = config["mask_file"]  # dacapo should know this
        mask_dataset = config["mask_dataset"]  # dacapo should know this
        db_name = config["db_name"]  # dacapo knows
        db_host = config["db_host"]  # dacapo knows
        queue = config["queue"]  # general multiprocessing argument
        context = config["context"]  # post_processor should know this
        num_voxels_in_block = config[
            "num_voxels_in_block"
        ]  # post_processor should know this
        fragments_in_xy = config["fragments_in_xy"]  # post_processor should know this
        epsilon_agglomerate = config[
            "epsilon_agglomerate"
        ]  # post_processor should know this (0 by default)
        filter_fragments = config["filter_fragments"]  # post_processor should know this
        replace_sections = config["replace_sections"]  # post_processor should know this # don't use this

        logger.info("Reading affs from %s", affs_file)
        affs = daisy.open_ds(affs_file, affs_dataset, mode="r")

        logger.info("Reading fragments from %s", fragments_file)
        fragments = daisy.open_ds(fragments_file, fragments_dataset, mode="r+")

        if mask_file:

            logger.info("Reading mask from %s", mask_file)
            mask = daisy.open_ds(mask_file, mask_dataset, mode="r")

        else:

            mask = None

        # open RAG DB
        logger.info("Opening RAG DB...")
        raise Exception("Mongo storage should be handled by dacapo.Store class")
        raise NotImplementedError("Hard coded position attrs")
        rag_provider = daisy.persistence.MongoDbGraphProvider(
            db_name,
            host=db_host,
            mode="r+",
            directed=False,
            position_attribute=["center_z", "center_y", "center_x"],
        )
        logger.info("RAG DB opened")

        # open block done DB
        raise Exception("Mongo storage should be handled by dacapo.Store class")
        client = pymongo.MongoClient(db_host)
        db = client[db_name]
        blocks_extracted = db["blocks_extracted"]

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
                replace_sections=replace_sections,
            )

            document = {
                "num_cpus": 5,
                "queue": queue,
                "block_id": block.block_id,
                "read_roi": (block.read_roi.get_begin(), block.read_roi.get_shape()),
                "write_roi": (block.write_roi.get_begin(), block.write_roi.get_shape()),
                "start": start,
                "duration": time.time() - start,
            }
            blocks_extracted.insert(document)

            client.release_block(block, ret=0)

    def blockwise_agglomerate_worker(self):
        """
        logging.info("Reading affs from %s", affs_file)
        affs = daisy.open_ds(affs_file, affs_dataset, mode='r')

        network_dir = os.path.join(experiment, setup, str(iteration), merge_function)

        logging.info("Reading fragments from %s", fragments_file)
        fragments = daisy.open_ds(fragments_file, fragments_dataset, mode='r')

        client = pymongo.MongoClient(db_host)
        db = client[db_name]

        blocks_agglomerated = ''.join([
            'blocks_agglomerated_',
            merge_function])

        if ''.join(['blocks_agglomerated_', merge_function]) not in db.list_collection_names():
            blocks_agglomerated = db[blocks_agglomerated]
            blocks_agglomerated.create_index(
                    [('block_id', pymongo.ASCENDING)],
                    name='block_id')
        else:
            blocks_agglomerated = db[blocks_agglomerated]

        context = daisy.Coordinate(context)
        total_roi = affs.roi.grow(context, context)

        # total_roi = daisy.Roi((0, 0, 67200), (900000, 285600, 403200))
        # total_roi = daisy.Roi((459960, 92120, 217952), (80040, 75880, 62048))

        # total_roi = daisy.Roi((50800, 43200, 44100), (10800, 10800, 10800))
        # total_roi = daisy.Roi((40000, 32400, 33300), (32400,)*3)
        # total_roi = daisy.Roi((96504,51660,44904),(1500,)*3)
        # total_roi = total_roi.grow(context, context)

        read_roi = daisy.Roi((0,)*affs.roi.dims(), block_size).grow(context, context)
        write_roi = daisy.Roi((0,)*affs.roi.dims(), block_size)

        daisy.run_blockwise(
            total_roi,
            read_roi,
            write_roi,
            process_function=lambda: start_worker(
                affs_file,
                affs_dataset,
                fragments_file,
                fragments_dataset,
                db_host,
                db_name,
                queue,
                merge_function,
                network_dir),
            check_function=lambda b: check_block(
                blocks_agglomerated,
                b),
            num_workers=num_workers,
            read_write_conflict=False,
            fit='shrink')
        """
        config = None

        affs_file = config["affs_file"]  # dacapo knows
        affs_dataset = config["affs_dataset"]  # dacapo knows
        fragments_file = config["fragments_file"]  # dacapo knows
        fragments_dataset = config["fragments_dataset"]  # dacapo knows
        db_host = config["db_host"]  # dacapo knows
        db_name = config["db_name"]  # dacapo knows
        queue = config["queue"]  # who knows?
        merge_function = config["merge_function"]  # watershed should know

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

        # open RAG DB
        raise NotImplementedError("Dacapo.Store should handle mongodb storage")
        logging.info("Opening RAG DB...")
        raise NotImplementedError("Hard coded position attrs")
        rag_provider = daisy.persistence.MongoDbGraphProvider(
            db_name,
            host=db_host,
            mode="r+",
            directed=False,
            edges_collection="edges_" + merge_function,
            position_attribute=["center_z", "center_y", "center_x"],
        )
        logging.info("RAG DB opened")

        # open block done DB
        raise NotImplementedError("Dacapo.Store should handle mongodb storage")
        client = pymongo.MongoClient(db_host)
        db = client[db_name]
        blocks_agglomerated = "".join(["blocks_agglomerated_", merge_function])

        blocks_agglomerated = db[blocks_agglomerated]

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

            document = {
                "num_cpus": 5,
                "queue": queue,
                "block_id": block.block_id,
                "read_roi": (block.read_roi.get_begin(), block.read_roi.get_shape()),
                "write_roi": (block.write_roi.get_begin(), block.write_roi.get_shape()),
                "start": start,
                "duration": time.time() - start,
            }
            blocks_agglomerated.insert(document)

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