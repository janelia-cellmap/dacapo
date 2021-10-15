from .datasets import Dataset

from abc import ABC, abstractmethod


class DataSplit(ABC):

    @property
    @abstractmethod
    def train(self) -> Dataset:
        """The Dataset to train on."""
        pass
