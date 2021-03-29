from .post_processor_abc import PostProcessorABC
from .steps import ArgMaxStep

import attr


@attr.s
class ArgMax(PostProcessorABC):
    outputs: str = attr.ib(
        default="labels", metadata={"help_text": "The name of the provided outputs."}
    )
    # steps: Tuple[ArgMaxStep] = attr.ib()

    def tasks(self, pred_id: str):
        """
        pred_id should be the unique id of the predictions you are post processing.
        i.e. run.id + iteration or prediction.id if run during prediction
        """
        tasks, parameters = ArgMaxStep().tasks(
            pred_id,
            container,
            probabilities_dataset,
            labels_dataset,
        )

        return tasks, parameters
