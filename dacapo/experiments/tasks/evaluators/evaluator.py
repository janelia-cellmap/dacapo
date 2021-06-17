from abc import ABC, abstractmethod


class Evaluator(ABC):
    """Base class of all evaluators.

    An evaluator takes a post-processor's output and compares it against
    ground-truth.
    """

    @abstractmethod
    def evaluate(
            self,
            output_container,
            output_dataset,
            gt_container,
            gt_dataset):
        """Compare an output dataset against ground-truth.

        Since the output is in general too large to be held in memory, this
        method receives the path to zarr containers and the dataset names of
        the output and ground-truth.

        Should return an instance of ``EvaluationScores``.
        """
        pass
