class EvaluationScores:
    """A class used represent the evaluation scores.

    This base class is used to provide an interface for different types of evaluation
    criteria. It provides abstractmethods for subclasses to implement specific evaluation
    criteria, their bounds and whether to store the best results.

    """

    @property
    @abstractmethod
    def criteria(self) -> List[str]:
        """Abstract method for criteria property
        
        This method should be overriden by subclasses to provide the evaluation criteria.
        
        Returns:
            List[str]: List of the evaluation criteria.
        """
        pass

    @staticmethod
    @abstractmethod
    def higher_is_better(criterion: str) -> bool:
        """
        Abstract method to check if higher is better for the given criterion.

        This method should be overriden by subclasses to provide the logic for determining
        whether higher scores are considered better for the provided criterion.

        Args: 
            criterion (str): The evaluation criterion.

        Returns:
            bool: True if higher scores are better, False otherwise.
        """
        pass

    @staticmethod
    @abstractmethod
    def bounds(criterion: str) -> Tuple[float, float]:
        """
        Abstract method to get the bounds for the given criterion.

        Subclasses should override this method to provide the lower and upper bounds for the
        provided criterion.

        Args: 
            criterion (str): The evaluation criterion.

        Returns:
            Tuple[float, float]: The lower and upper bounds for the criterion.
        """
        pass

    @staticmethod
    @abstractmethod
    def store_best(criterion: str) -> bool:
        """
        Abstract method to check if the best results should be saved.

        Subclasses should override this method to specify whether the best validation block
        and model weights should be saved for the provided criterion.

        Args:
            criterion (str): The evaluation criterion.

        Returns:
            bool: True if the best results should be saved, False otherwise.
        """
        pass