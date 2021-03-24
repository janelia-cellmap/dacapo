import attr
import daisy
from daisy import Task, Block
from daisy.persistence import MongoDbGraphProvider
import numpy as np
from funlib.segment.arrays import replace_values

from .step_abc import PostProcessingStepABC
from dacapo.store import MongoDbStore

from typing import List, Optional
import logging
import time

logger = logging.getLogger(__name__)


@attr.s
class Segment(PostProcessingStepABC):
    step_id: str = attr.ib(default="segment")

    # blockwise_processing_parameters
    write_shape: Optional[List[int]] = attr.ib(default=None)
    context: Optional[List[int]] = attr.ib(default=None)
    num_workers: int = attr.ib(default=2)

    def tasks(
        self,
        pred_id,
        container,
        fragments_dataset,
        segmentation_dataset,
        lookup,
        upstream_tasks=None,
    ):
        if upstream_tasks is None:
            upstream_tasks = [None, {}]
        tasks, task_parameters = [], []
        for upstream_task, upstream_parameters in zip(*upstream_tasks):
            # expand existing parameters with "merge_function"
            parameters = dict(**upstream_parameters)

            # open dataset
            frags = daisy.open_ds(container, fragments_dataset)

            # input_roi defined by provided dataset
            # TODO: allow for subrois?
            input_roi = frags.roi
            if self.write_shape is None:
                # default to 128 per axis or input_roi shape if less than that
                write_shape = daisy.Coordinate(
                    tuple(
                        min(a, b)
                        for a, b in zip(input_roi.shape, [128] * input_roi.dims)
                    )
                )
            else:
                write_shape = self.write_shape

            # get context
            # TODO: do we need context for agglomeration?
            context = daisy.Coordinate((0,) * input_roi.dims)

            # define block read/write rois based on write_shape and context
            read_roi = daisy.Roi((0,) * context.dims, write_shape + context * 2)
            write_roi = daisy.Roi(context, write_shape)

            task = Task(
                task_id=f"{pred_id}_{self.step_id}",
                total_roi=input_roi,
                read_roi=read_roi,
                write_roi=write_roi,
                process_function=self.get_process_function(
                    pred_id=pred_id,
                    container=container,
                    fragments_dataset=fragments_dataset,
                    segmentation_dataset=segmentation_dataset,
                    lookup=lookup,
                ),
                check_function=self.get_check_function(pred_id),
                num_workers=self.num_workers,
                fit="overhang",
            )
            tasks.append(task)
            task_parameters.append(parameters)

        return tasks, task_parameters

    def get_process_function(
        self,
        pred_id,
        container,
        fragments_dataset,
        segmentation_dataset,
        lookup,
    ):
        store = MongoDbStore()
        rag_provider = MongoDbGraphProvider(
            store.db_name,
            host=store.db_host,
            mode="r+",
            directed=False,
            nodes_collection=f"{pred_id}_{self.step_id}_frags",
            edges_collection=f"{pred_id}_{self.step_id}_frag_agglom",
            position_attribute=["center_z", "center_y", "center_x"],
        )
        # TODO: fragments/segmentations dataset will depend on the upstream task since
        # they will be produced with different parameters
        fragments = daisy.open_ds(container, fragments_dataset, mode="r")
        segmentation = daisy.open_ds(container, segmentation_dataset, mode="r+")

        def process_block(b: Block):
            start = time.time()
            block_fragments = fragments.to_ndarray(b.write_roi)
            logger.info("%.3fs" % (time.time() - start))

            start = time.time()

            relabelled = np.zeros_like(block_fragments)

            agg_graph = rag_provider.get_graph(roi=b.write_roi)

            lut = np.array(
                [
                    (np.uint64(node), np.uint64(attrs[lookup]))
                    for node, attrs in agg_graph.nodes.items()
                    if lookup in attrs
                ]
            )

            replace_values(block_fragments, lut[:, 0], lut[:, 1], out_array=relabelled)
            logger.info("%.3fs" % (time.time() - start))

            segmentation[b.write_roi] = relabelled

            store.mark_block_done(
                pred_id, self.step_id, b.block_id, start, time.time() - start
            )

        return process_block
