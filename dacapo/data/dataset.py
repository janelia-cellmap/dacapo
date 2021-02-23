import itertools

from .group import TrainValidateSplit, TestSplit, ArrayGroup, GraphGroup

from typing import List, Dict, Optional
import attr

from .data_source import DataSource
from .keys import ArrayKey, GraphKey, DataKey
from dacapo.data.padding import PaddingOption


@attr.s
class Dataset:
    name: str = attr.ib()
    arrays: List[ArrayKey] = attr.ib()
    graphs: List[GraphKey] = attr.ib()
    train_padding: PaddingOption = attr.ib(
        default=PaddingOption.VALID,
    )
    validation_padding: PaddingOption = attr.ib(
        default=PaddingOption.VALID,
    )
    predict_padding: PaddingOption = attr.ib(
        default=PaddingOption.VALID,
    )
    train_sources: Optional[Dict[DataKey, DataSource]] = attr.ib()
    validate_sources: Optional[Dict[DataKey, DataSource]] = attr.ib()
    predict_sources: Optional[Dict[DataKey, DataSource]] = attr.ib()

    def __attrs_post_init__(self):
        # add attrs for each array and graph key
        for array in self.arrays:
            self.__setattr__(array, TrainValidateSplit(array))
        for graph in self.graphs:
            self.__setattr__(graph, TrainValidateSplit(graph))

        for data_key in itertools.chain(self.arrays, self.graphs):
            getattr(self, data_key).train = self.train_sources[data_key]
            getattr(self, data_key).validate = self.validate_sources[data_key]