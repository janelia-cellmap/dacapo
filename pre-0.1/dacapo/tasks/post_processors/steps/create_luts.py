import attr
from daisy import Task, Block
from daisy.persistence import MongoDbGraphProvider
import numpy as np
from funlib.segment.graphs.impl import connected_components

from .step_abc import PostProcessingStepABC
from dacapo.store import MongoDbStore

from typing import List
import time
import logging

logger = logging.getLogger(__name__)


@attr.s
class CreateLUTS(PostProcessingStepABC):
    step_id: str = attr.ib(default="luts")
    # grid searchable arguments
    threshold: List[float] = attr.ib(factory=list)

    def tasks(
        self,
        pred_id,
        roi,
        upstream_tasks=None,
    ):
        """
        lookup: the node attribute name upon which we store the final id
        """

        if upstream_tasks is None:
            upstream_tasks = [None, {}]
        tasks, task_parameters = [], []
        for i, threshold in enumerate(self.merge_function):
            for upstream_task, upstream_parameters in zip(*upstream_tasks):
                parameters = dict(**upstream_parameters)
                parameters["threshold"] = threshold

                upstream_tasks = []
                if upstream_task is not None:
                    upstream_tasks.append(upstream_task)

                task = Task(
                    task_id=f"{pred_id}_{self.step_id}",
                    total_roi=roi,
                    read_roi=roi,
                    write_roi=roi,
                    process_function=self.get_process_function(
                        pred_id=pred_id,
                        threshold=threshold,
                    ),
                    check_function=self.get_check_function(pred_id),
                    num_workers=1,
                    upstream_tasks=upstream_tasks,
                )
                tasks.append(task)
                task_parameters.append(parameters)

        return tasks, task_parameters

    def get_process_function(
        self,
        pred_id,
        threshold,
    ):
        store = MongoDbStore()
        # TODO: Depends on parameters
        logger.error(
            "'lookup' should depend on the specific upstream parameter set and "
            "chosen threshold to avoid conflicts in mongodb."
        )
        lookup = f"{pred_id}_{self.step_id}"
        rag_provider = MongoDbGraphProvider(
            store.db_name,
            host=store.db_host,
            mode="r+",
            directed=False,
            nodes_collection=f"{pred_id}_{self.step_id}_frags",  # TODO: depends on params
            edges_collection=f"{pred_id}_{self.step_id}_frag_agglom",
            position_attribute=["center_z", "center_y", "center_x"],
        )

        def process_block(b: Block):
            start = time.time()
            g = rag_provider.get_graph(b.roi)

            logger.info("Read graph in %.3fs" % (time.time() - start))

            assert g.number_of_nodes() > 0, f"No nodes found in roi {b.roi}"

            nodes = np.array(list(g.nodes()))
            edges = np.array([(u, v) for u, v in g.edges()], dtype=np.uint64)
            scores = np.array(
                [attrs["merge_score"] for edge, attrs in g.edges.items()],
                dtype=np.float32,
            )
            scores = np.nan_to_num(scores, nan=1)

            logger.debug(
                f"percentiles (1, 5, 50, 95, 99): {np.percentile(scores, [1,5,50,95,99])}"
            )

            logger.debug("Nodes dtype: ", nodes.dtype)
            logger.debug("edges dtype: ", edges.dtype)
            logger.debug("scores dtype: ", scores.dtype)

            logger.debug(
                "Complete RAG contains %d nodes, %d edges" % (len(nodes), len(edges))
            )

            start = time.time()

            logger.debug("Getting CCs for threshold %.3f..." % threshold)
            start = time.time()
            components = connected_components(nodes, edges, scores, threshold).astype(
                np.int64
            )
            logger.debug("%.3fs" % (time.time() - start))

            logger.debug(
                "Creating fragment-segment LUT for threshold %.3f..." % threshold
            )
            start = time.time()
            lut = np.array([(n, c) for n, c in zip(nodes, components)], dtype=int)

            logger.info("%.3fs" % (time.time() - start))

            logger.info(
                "Storing fragment-segment LUT for threshold %.3f..." % threshold
            )
            start = time.time()

            for node, component in lut:
                g.nodes[node][lookup] = int(component)

            g.update_node_attrs(attributes=[lookup])

            logger.info("%.3fs" % (time.time() - start))

            logger.info(
                "Created and stored lookup tables in %.3fs" % (time.time() - start)
            )

            store.mark_block_done(
                pred_id, self.step_id, b.block_id, start, time.time() - start
            )
            pass

        return process_block
