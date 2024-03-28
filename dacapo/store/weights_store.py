from dacapo.experiments.run import Run

import torch

from abc import ABC, abstractmethod
from typing import Optional
from collections import OrderedDict


class Weights:
    """
    A class representing the weights of a model and optimizer.

    Attributes:
        optimizer (OrderedDict[str, torch.Tensor]): The optimizer's state dictionary.
        model (OrderedDict[str, torch.Tensor]): The model's state dictionary.
    Methods:
        __init__(model_state_dict, optimizer_state_dict): Initializes the Weights object with the given model and optimizer state dictionaries.
    """

    optimizer: OrderedDict[str, torch.Tensor]
    model: OrderedDict[str, torch.Tensor]

    def __init__(self, model_state_dict, optimizer_state_dict):
        """
        Initializes the Weights object with the given model and optimizer state dictionaries.

        Args:
            model_state_dict (OrderedDict[str, torch.Tensor]): The state dictionary of the model.
            optimizer_state_dict (OrderedDict[str, torch.Tensor]): The state dictionary of the optimizer.
        Examples:
            >>> model = torch.nn.Linear(2, 2)
            >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            >>> weights = Weights(model.state_dict(), optimizer.state_dict())
        """
        self.model = model_state_dict
        self.optimizer = optimizer_state_dict


class WeightsStore(ABC):
    """
    Base class for network weight stores.

    Methods:
        load_weights(run, iteration): Load the weights of the given iteration into the given run.
        load_best(run, dataset, criterion): Load the best weights for the given run, dataset, and criterion into the given run.
        latest_iteration(run): Return the latest iteration for which weights are available for the given run.
        store_weights(run, iteration): Store the network weights of the given run.
        retrieve_weights(run, iteration): Retrieve the network weights of the given run.
        remove(run, iteration): Delete the weights associated with a specific run/iteration.
        retrieve_best(run, dataset, criterion): Retrieve the best weights for the given run, dataset, and criterion.
    """

    def load_weights(self, run: Run, iteration: int) -> None:
        """
        Load this iterations weights into the given run.
        Args:
            run (Run): The run to load the weights into.
            iteration (int): The iteration to load the weights from.
        Raises:
            ValueError: If the iteration is not available.
        Examples:
            >>> store = WeightsStore()
            >>> run = Run()
            >>> iteration = 0
            >>> store.load_weights(run, iteration)
        """
        weights = self.retrieve_weights(run.name, iteration)
        run.model.load_state_dict(weights.model)
        run.optimizer.load_state_dict(weights.optimizer)

    def load_best(self, run: Run, dataset: str, criterion: str) -> None:
        """
        Load the best weights for this Run,dataset,criterion into Run.model

        Args:
            run (Run): The run to load the weights into.
            dataset (str): The dataset to load the weights from.
            criterion (str): The criterion to load the weights from.
        Raises:
            ValueError: If the best iteration is not available.
        Examples:
            >>> store = WeightsStore()
            >>> run = Run()
            >>> dataset = 'mnist'
            >>> criterion = 'accuracy'
            >>> store.load_best(run, dataset, criterion)

        """
        best_iteration = self.retrieve_best(run.name, dataset, criterion)
        self.load_weights(run, best_iteration)

    @abstractmethod
    def latest_iteration(self, run: str) -> Optional[int]:
        """
        Return the latest iteration for which weights are available for the
        given run.

        Args:
            run (str): The name of the run.
        Returns:
            int: The latest iteration for which weights are available.
        Raises:
            ValueError: If no weights are available for the given run.
        Examples:
            >>> store = WeightsStore()
            >>> run = 'run_0'
            >>> store.latest_iteration(run)
        """
        pass

    @abstractmethod
    def store_weights(self, run: Run, iteration: int) -> None:
        """
        Store the network weights of the given run.

        Args:
            run (Run): The run to store the weights of.
            iteration (int): The iteration to store the weights for.
        Raises:
            ValueError: If the iteration is already stored.
        Examples:
            >>> store = WeightsStore()
            >>> run = Run()
            >>> iteration = 0
            >>> store.store_weights(run, iteration)
        """
        pass

    @abstractmethod
    def retrieve_weights(self, run: str, iteration: int) -> Weights:
        """
        Retrieve the network weights of the given run.

        Args:
            run (str): The name of the run.
            iteration (int): The iteration to retrieve the weights for.
        Returns:
            Weights: The weights of the given run and iteration.
        Raises:
            ValueError: If the weights are not available.
        Examples:
            >>> store = WeightsStore()
            >>> run = 'run_0'
            >>> iteration = 0
            >>> store.retrieve_weights(run, iteration)
        """
        pass

    @abstractmethod
    def remove(self, run: str, iteration: int) -> None:
        """
        Delete the weights associated with a specific run/iteration

        Args:
            run (str): The name of the run.
            iteration (int): The iteration to delete the weights for.
        Raises:
            ValueError: If the weights are not available.
        Examples:
            >>> store = WeightsStore()
            >>> run = 'run_0'
            >>> iteration = 0
            >>> store.remove(run, iteration)

        """
        pass

    @abstractmethod
    def retrieve_best(self, run: str, dataset: str, criterion: str) -> int:
        """
        Retrieve the best weights for this run/dataset/criterion

        Args:
            run (str): The name of the run.
            dataset (str): The dataset to retrieve the best weights for.
            criterion (str): The criterion to retrieve the best weights for.
        Returns:
            int: The iteration of the best weights.
        Raises:
            ValueError: If the best weights are not available.
        Examples:
            >>> store = WeightsStore()
            >>> run = 'run_0'
            >>> dataset = 'mnist'
            >>> criterion = 'accuracy'
            >>> store.retrieve_best(run, dataset, criterion)
        """
        pass
