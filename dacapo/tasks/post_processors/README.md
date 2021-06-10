# PostProcessors

To add support for a new PostProcessor, you must create an `attr.s` class that
subclasses the `PostProcessorABC` like this:

```python
from .post_processor_abc import PostProcessorABC
from .steps import MyStep1, MyStep2

import attr


@attr.s
class MyPostprocessor(PostProcessorABC):
    name: str = attr.ib(default="MyPostprocessor")

    step1_parameter: List[float] = attr.ib(factory=list)
    step2_parameter: List[int] = attr.ib(factory=list)

    def tasks(self, pred_id, container, in_dataset, out_dataset):
        """
        pred_id: should be the unique id of the predictions you are post processing.
            i.e. f"{run.id}_validation_{iteration}" if run during validation

        container: The zarr container where the data to postprocess exists, and
            where the post processed data will be written. Any intermediate
            data will also be written here

        in_dataset: The dataset in which to find data for postprocessing. 
            Usually "volumes/{predictor_name}"

        out_dataset: The dataset in which to store the post_processed data.
            Usually "volumes/{predictor_name}_{post_processor_name}_{i}", for
            i, parameters in enumerate(parameters).
        """
        half_way_dataset = "volumes/half_way"
        tasks, parameters = MyStep1(step1_parameter=self.step1_parameter).tasks(
            pred_id, container, in_dataset, half_way_dataset
        )
        tasks, parameters = MyStep2(step2_parameter=self.step2_parameter).tasks(
            pred_id, container, half_way_dataset, out_dataset,
            upstream_tasks=(tasks, parameters),
        )

        return tasks, parameters

```

The only required function is `tasks`, that returns a list of daisy tasks,
along with a list of parameters that were used to generate the output. These
tasks are all assumed to be cpu only tasks.

Once you have added support for a new PostProcessor, you must:
1) import it into `__init__.py`
2) include it in the `AnyPostProcessor` Union type.
3) Add it to the list of exposed configurable types in `dacapo.configurables`. This Allows
the dacapo-dashboard user interface to read the parameters with their types and metadata.