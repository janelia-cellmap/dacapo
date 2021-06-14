# PostProcessorStep

To add support for a new PostProcessor, you must create an `attr.s` class that
subclasses the `PostProcessorStepABC` like this:

```python
import attr
import numpy as np
import daisy

from dacapo.store import MongoDbStore
from .step_abc import PostProcessingStepABC

import time
from typing import Optional, List


@attr.s
class MyStep(PostProcessingStepABC):
    step_id: str = attr.ib(default="my_new_step")

    # parameter ranges:
    my_parameter: List[float] = attr.ib(factory=list)

    # blockwise_processing_parameters
    write_shape: Optional[List[int]] = attr.ib(default=None)
    context: Optional[List[int]] = attr.ib(default=None)
    num_workers: int = attr.ib(default=2)
    fit: str = attr.ib(default="overhang")

    def tasks(
        self,
        pred_id: str,
        *args,
        **kwargs,
        upstream_tasks=None,
    ):
        """
        pred_id: should be the unique id of the predictions you are post processing.
            i.e. f"{run.id}_validation_{iteration}" if run during validation

        upstream_tasks: an optional input of form Tuple[List[Task],List[Dict]],
            where each task has its associated set of parameters that were used
            to create that task.
        """

        tasks, task_parameters = [], []

        if upstream_tasks is None:
            upstream_tasks = [None, {}]

        for upstream_task, upstream_parameters in zip(*upstream_tasks):
            for i, my_param in enumerate(self.my_parameters):
                parameters = dict(**upstream_parameters)
                parameters[f"{self.step_id}_my_parameter"] = self.my_parameter

                task_input_dataset = parameters[input_dataset]
                input_daisy_array = daisy.open_ds(container, task_input_dataset, mode="r")

                # input_roi defined by provided dataset
                input_roi = input_daisy_array.roi

                # get write_shape
                if self.write_shape is None:
                    # Sensible default defined here. This way we can be agnostic
                    # to input_roi dimensions
                    write_shape = daisy.Coordinate(
                        tuple(
                            min(a, b)
                            for a, b in zip(input_roi.shape, [128] * input_roi.dims)
                        )
                    )
                else:
                    write_shape = self.write_shape

                # get context
                if self.context is None:
                    # Sensible default defined here. This way we can be agnostic
                    # to input_roi dimensions
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

                task_id = f"{pred_id}_{self.step_id}_{i}"

                t = daisy.Task(
                    task_id=task_id,
                    total_roi=input_roi,
                    read_roi=read_roi,
                    write_roi=write_roi,
                    process_function=self.get_process_function(pred_id, *args, **kwargs),
                    check_function=self.get_check_function(pred_id, i),
                    num_workers=self.num_workers,
                    fit=self.fit,
                )
                tasks.append(t)
                task_parameters.append(parameters)

        return tasks, task_parameters

    def get_process_function(self, pred_id, *args, i=None, **kwargs):
        store = MongoDbStore()

        def process_block(b):
            start = time.time()

            # process your block and write it out to file

            store.mark_block_done(
                pred_id, self.step_id, b.block_id, start, time.time() - start
            )

        return process_block


```

Each `PostProcessorStep` must define a `tasks` method. This method takes a "prediction_id"
argument and an optional "upstream_tasks" argument. It can also have arbitrary other arguments.
The `tasks` method must define a processor for each value in its parameter ranges, for each
upstream task, and return them.
Every `PostProcessorStep` must also define a `get_process_function` that takes the "pred_id",
an optional "i" argument for the parameter set value, and arbitrary other arguments. It must
return a function that takes a block and processes it.

Once you have added support for a new PostProcessorStep, you must:
1) import it into `__init__.py`
2) include it in the `AnyPostProcessorStep` Union type.
3) Add it to the list of exposed configurable types in `dacapo.configurables`. This Allows
the dacapo-dashboard user interface to read the parameters with their types and metadata.