import attr
import daisy
from daisy import Task, Block
from daisy.persistence import MongoDbGraphProvider
import lsd

from .step_abc import PostProcessingStepABC
from dacapo.store import MongoDbStore
from .waterz_merge_functions import MergeFunction

from typing import List, Optional
import time


@attr.s
class Agglomerate(PostProcessingStepABC):
    step_id: str = attr.ib(default="agglomerate")
    # grid searchable arguments
    merge_function: List[MergeFunction] = attr.ib()

    # blockwise_processing_parameters
    write_shape: Optional[List[int]] = attr.ib(default=None)
    context: Optional[List[int]] = attr.ib(default=None)
    num_workers: int = attr.ib(default=2)

    def tasks(
        self,
        pred_id,
        container,
        affs_dataset,
        fragments_dataset,
        upstream_tasks=None,
    ):
        if upstream_tasks is None:
            upstream_tasks = [None, {}]
        tasks, task_parameters = [], []
        for i, merge_func in enumerate(self.merge_function):
            for upstream_task, upstream_parameters in zip(*upstream_tasks):
                # expand existing parameters with "merge_function"
                parameters = dict(**upstream_parameters)
                parameters["merge_function"] = merge_func

                # open dataset
                affs = daisy.open_ds(container, affs_dataset)

                # input_roi defined by provided dataset
                # TODO: allow for subrois?
                input_roi = affs.roi
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
                if self.context is None:
                    # default to 20 per axis or input_roi shape if less than that
                    context = daisy.Coordinate(
                        tuple(
                            min(a, b)
                            for a, b in zip(input_roi.shape, [20] * input_roi.dims)
                        )
                    )
                else:
                    context = self.context

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
                        affs_dataset=affs_dataset,
                        fragments_dataset=fragments_dataset,
                        merge_function=merge_func,
                    ),
                    check_function=self.get_check_function,
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
        affs_dataset,
        fragments_dataset,
        merge_function,
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
        affs = daisy.open_ds(container, affs_dataset, mode="r")
        # TODO: fragments dataset will depend on the upstream task since
        # they will be produced with different parameters
        fragments = daisy.open_ds(container, fragments_dataset, mode="r")

        def process_block(b: Block):
            start = time.time()
            lsd.agglomerate_in_block(
                affs,
                fragments,
                rag_provider,
                b,
                merge_function=merge_function,
                threshold=1.0,
            )

            store.mark_block_done(
                pred_id, self.step_id, b.block_id, start, time.time() - start
            )
            pass

        return process_block

    def get_check_function(self, task_id):
        def check_block(b: Block):
            pass

        return check_block
