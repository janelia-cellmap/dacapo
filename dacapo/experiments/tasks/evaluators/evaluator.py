from abc import ABC, abstractmethod


class Evaluator(ABC):
    """Base class of all evaluators.

    An evaluator takes a post-processor's output and compares it against
    ground-truth.
    """

    @abstractmethod
    def evaluate(
            self,
            output_array,
            evaluation_dataset):
        """Compare an output dataset against ground-truth from an evaluation
        dataset.

        The evaluation dataset is a dictionary mapping from ``DataKey`` to
        ``ArraySource`` or ``GraphSource``.

        Should return an instance of ``EvaluationScores``.
        """
        pass
