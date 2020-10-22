from gunpowder import ArrayKey, GraphKey

from typing import Union, Iterable, List

from .dataset import Dataset, ArrayDataset, GraphDataset


class ArrayGroup:
    """
    A group of array datasets that all represent samples of the
    same category. i.e. you might have 2 blocks of raw and
    gt data to train on.
    """

    def __init__(self, datasets: List[ArrayDataset]):
        self._datasets = datasets

    @property
    def datasets(self):
        return self._datasets

    @property
    def axes(self):
        """
        All members should have the same axes
        """
        return self._datasets[0].axes

    def get_sources(self, *output_keys: ArrayKey):
        """
        A dacapo dataset is expected to provide a list of gunpowder trees
        that all provide the given output keys.
        """
        return [d.get_source() for d in self.datasets]

    def __getattr__(self, attr):
        if hasattr(self, attr):
            getattr(self, attr)
        else:
            return getattr(self.datasets[0], attr)


class GraphGroup:
    """
    A group of graph datasets that all represent samples of the
    same category.
    """

    def __init__(self, datasets: List[GraphDataset]):
        self._datasets = datasets

    @property
    def datasets(self):
        return self._datasets

    def get_sources(self, *output_keys: GraphKey):
        """
        A dacapo dataset is expected to provide a list of gunpowder trees
        that all provide the given output keys.
        """
        return [d.get_source() for d in self.datasets]

    def __getattr__(self, attr):
        if hasattr(self, attr):
            getattr(self, attr)
        else:
            return getattr(self.datasets[0], attr)


class TrainValidateSplit:
    """
    A train/validate data split. Dacapo always expects your data
    to be split into train and validate data.

    These are grouped here. It is expected that many attributes
    between train and validate will stay consistent. i.e. axes, num_channels,
    etc. Thus if you try to get an attribute from the group via
    group.num_channels, it simply gets group.train.num_channels.

    If you specifically want an attribute from the validation dataset
    you must explicitly ask for group.valiate.attribute
    """

    def __init__(self, name: str):
        self._name = name
        self._train = None
        self._validate = None

    @property
    def name(self):
        return self._name

    @property
    def train(self):
        if self._train is None:
            raise Exception(f"{self.name} train/validation group has no training data")
        return self._train

    @train.setter
    def train(self, dataset: Union[Dataset, Iterable[Dataset]]):
        if isinstance(dataset, Iterable):
            self._train = dataset
        else:
            self._train = dataset

    @property
    def validate(self):
        if self._validate is None:
            raise Exception(
                f"{self.name} train/validation group has no validation data"
            )
        return self._validate

    @validate.setter
    def validate(self, dataset: Union[Dataset, Iterable[Dataset]]):
        if isinstance(dataset, Iterable):
            self._validate = dataset
        else:
            self._validate = dataset

    def __getattr__(self, attr):
        if attr in self.__dict__:
            self.__dict__[attr]
        else:
            return getattr(self.train, attr)
