class Weights:
    """
    This is a class for handling weights for the model's state and optimizer's state.
    
    Attributes:
        optimizer (OrderedDict[str, torch.Tensor]): The weights tensor for optimizer's state.
        model (OrderedDict[str, torch.Tensor]): The weights tensor for model's state.
    """

    def __init__(self, model_state_dict, optimizer_state_dict):
        """
        Initializes an instance of Weights.
        
        Args:
            model_state_dict (OrderedDict): The state_dict of the model.
            optimizer_state_dict (OrderedDict): The state_dict of the optimizer.
        """
        self.model = model_state_dict
        self.optimizer = optimizer_state_dict


class WeightsStore(ABC):
    """
    This is an abstract base class (ABC) for handling operations related to the 
    storage of network weights.

    It defines some common methods that every derived class should implement.
    """

    def load_weights(self, run: Run, iteration: int) -> None:
        """
        Loads model and optimizer weights from a given iteration into a run instance.

        This method does not return anything.

        Args:
            run (Run): The Run instance to load weights into.
            iteration (int): The iteration from which to load the weights.
        """
        weights = self.retrieve_weights(run.name, iteration)
        run.model.load_state_dict(weights.model)
        run.optimizer.load_state_dict(weights.optimizer)

    def load_best(self, run: Run, dataset: str, criterion: str) -> None:
        """
        Loads the best weights for a specific run, dataset, and criterion into a run instance.

        This method does not return anything.

        Args:
            run (Run): The Run instance to load best weights into.
            dataset (str): The dataset associated with the best weights.
            criterion (str): The criterion associated with the best weights.
        """
        best_iteration = self.retrieve_best(run.name, dataset, criterion)
        self.load_weights(run, best_iteration)

    @abstractmethod
    def latest_iteration(self, run: str) -> Optional[int]:
        """
        An abstract method that is expected to return the latest iteration for 
        which weights are available for a given run.

        Args:
            run (str): The name of the run.

        Returns:
            int, optional: The latest iteration, or None if not available.
        """
        pass

    @abstractmethod
    def store_weights(self, run: Run, iteration: int) -> None:
        """
        An abstract method that is expected to store the weights of the given run at a 
        specific iteration.

        This method does not return anything.

        Args:
            run (Run): The Run instance whose weights are to be stored.
            iteration (int): The iteration at which to store the weights.
        """
        pass

    @abstractmethod
    def retrieve_weights(self, run: str, iteration: int) -> Weights:
        """
        An abstract method that is expected to return the Weights object of the given run
        at a specific iteration.

        Args:
            run (str): The name of the run.
            iteration (int): The iteration from which to retrieve the weights.

        Returns:
            Weights: A Weights object containing the model and optimizer weights.
        """
        pass

    @abstractmethod
    def remove(self, run: str, iteration: int) -> None:
        """
        An abstract method that is expected to remove the weights of the given run at a 
        specific iteration.

        This method does not return anything.

        Args:
            run (str): The name of the run.
            iteration (int): The iteration from which to remove the weights.
        """
        pass

    @abstractmethod
    def retrieve_best(self, run: str, dataset: str, criterion: str) -> int:
        """
        An abstract method that is expected to retrieve the best weights for the given 
        run, dataset, and criterion.

        Args:
            run (str): The name of the run.
            dataset (str): The dataset associated with the best weights.
            criterion (str): The criterion associated with the best weights.

        Returns:
            int: The iteration at which the best weights occur.
        """
        pass