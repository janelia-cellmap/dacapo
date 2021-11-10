from .datasets import Dataset

from abc import ABC, abstractmethod
from typing import List, Optional


class DataSplit(ABC):
    @property
    @abstractmethod
    def train(self) -> List[Dataset]:
        """The Dataset to train on."""
        pass

    @property
    def validate(self) -> Optional[List[Dataset]]:
        return None
