import attr
import daisy
from daisy import Task, Block
from daisy.persistence import MongoDbGraphProvider

from .step_abc import PostProcessingStepABC
from .watershed_helpers import watershed_in_block
from dacapo.store import MongoDbStore

from typing import List, Optional
import itertools


@attr.s
class Fragment(PostProcessingStepABC):
    step_id: str = attr.ib(default="fragment")

    # grid searchable arguments
    filter_fragments: List[float] = attr.ib(factory=lambda: list([0]))
    fragments_in_xy: List[bool] = attr.ib(factory=lambda: list([False]))
    epsilon_agglomerate: List[float] = attr.ib(factory=lambda: list([0]))
    min_seed_distance: List[float] = attr.ib(factory=lambda: list([1]))
    compactness: List[float] = attr.ib(factory=lambda: list([0]))

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
        mask_file=None,
        mask_dataset=None,
        upstream_tasks=None,
    ):
        # handle upstream tasks: Tuple[List[Task], List[Dict]]

        store = MongoDbStore()
        tasks, task_parameters = [], []

        if upstream_tasks is None:
            upstream_tasks = [None, {}]

        for upstream_task, upstream_parameters in upstream_tasks:
            for i, (
                filter_fragments,
                fragments_in_xy,
                epsilon_agglomerate,
                min_seed_distance,
                compactness,
            ) in enumerate(
                itertools.product(
                    self.filter_fragments,
                    self.fragments_in_xy,
                    self.epsilon_agglomerate,
                    self.min_seed_distance,
                    self.compactness,
                )
            ):
                parameters = dict(**upstream_parameters)
                new_parameters = {
                    "filter_fragments": filter_fragments,
                    "fragments_in_xy": fragments_in_xy,
                    "epsilon_agglomerate": epsilon_agglomerate,
                    "min_seed_distance": min_seed_distance,
                    "compactness": compactness,
                }
                parameters.update(new_parameters)

                task_id = f"{pred_id}_{self.step_id}_{i}"

                affs = daisy.open_ds(container, affs_dataset, "r")
                input_roi = affs.roi

                # get write_shape
                if self.write_shape is None:
                    # default to 512 per axis or input_roi shape if less than that
                    write_shape = daisy.Coordinate(
                        tuple(
                            min(a, b)
                            for a, b in zip(input_roi.shape, [512] * input_roi.dims)
                        )
                    )
                else:
                    write_shape = self.write_shape

                # get context
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

                rag_provider = MongoDbGraphProvider(
                    store.db_name,
                    host=store.db_host,
                    nodes_collection=f"{task_id}_nodes",
                    edges_collection=f"{task_id}_edges",
                    mode="r+",
                    directed=False,
                    position_attribute=["center_z", "center_y", "center_x"],
                )
                affs = daisy.open_ds(container, affs_dataset, mode="r")
                if mask_file is not None and mask_dataset is not None:
                    mask = daisy.open_ds(mask_file, mask_dataset, mode="r")
                else:
                    mask = None

                fragments = daisy.open_ds(container, fragments_dataset, mode="r+")

                upstream_tasks = []
                if upstream_task is not None:
                    upstream_tasks.append(upstream_task)

                task = Task(
                    task_id=task_id,
                    total_roi=input_roi,
                    read_roi=read_roi,
                    write_roi=write_roi,
                    process_function=self.get_process_function(
                        filter_fragments=filter_fragments,
                        fragments_in_xy=fragments_in_xy,
                        epsilon_agglomerate=epsilon_agglomerate,
                        min_seed_distance=min_seed_distance,
                        compactness=compactness,
                        task_id=task_id,
                        rag_provider=rag_provider,
                        affs=affs,
                        fragments=fragments,
                        mask=mask,
                    ),
                    check_function=self.get_check_function(pred_id),
                    num_workers=self.num_workers,
                    fit="overhang",
                    upstream_tasks=upstream_tasks,
                )

                tasks.append(task)
                task_parameters.append(parameters)

        return tasks, parameters

    def get_process_function(
        self,
        filter_fragments,
        fragments_in_xy,
        epsilon_agglomerate,
        min_seed_distance,
        compactness,
        task_id,
        rag_provider,
        affs,
        fragments,
        mask=None,
    ):
        def process_block(b: Block):
            context = (b.read_roi.shape - b.write_roi.shape) / 2
            num_voxels_in_block = b.write_roi.size / affs.voxel_size.size
            watershed_in_block(
                affs,
                b,
                context,
                rag_provider,
                fragments,
                num_voxels_in_block=int(num_voxels_in_block),
                mask=mask,
                fragments_in_xy=fragments_in_xy,
                epsilon_agglomerate=epsilon_agglomerate,
                filter_fragments=filter_fragments,
            )

        return process_block
