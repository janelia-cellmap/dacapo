from dacapo.experiments.run import Run

import torch

from abc import ABC, abstractmethod
from typing import Optional
from collections import OrderedDict


class Weights:
    

    optimizer: OrderedDict[str, torch.Tensor]
    model: OrderedDict[str, torch.Tensor]

    def __init__(self, model_state_dict, optimizer_state_dict):
        
        self.model = model_state_dict
        self.optimizer = optimizer_state_dict


class WeightsStore(ABC):
    

    def load_weights(self, run: Run, iteration: int) -> None:
        
        weights = self.retrieve_weights(run.name, iteration)
        run.model.load_state_dict(weights.model)
        run.optimizer.load_state_dict(weights.optimizer)

    def load_best(self, run: Run, dataset: str, criterion: str) -> None:
        
        best_iteration = self.retrieve_best(run.name, dataset, criterion)
        self.load_weights(run, best_iteration)

    @abstractmethod
    def latest_iteration(self, run: str) -> Optional[int]:
        
        pass

    @abstractmethod
    def store_weights(self, run: Run, iteration: int) -> None:
        
        pass

    @abstractmethod
    def retrieve_weights(self, run: str, iteration: int) -> Weights:
        
        pass

    @abstractmethod
    def remove(self, run: str, iteration: int) -> None:
        
        pass

    @abstractmethod
    def retrieve_best(self, run: str, dataset: str, criterion: str) -> int:
        
        pass
