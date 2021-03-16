from .data_source import DataSource, ArrayDataSource, GraphDataSource
from .keys import ArrayKey, GraphKey, DataKey
from dacapo.padding import PaddingOption
from dacapo.converter import converter

import attr

import itertools
from typing import List, Dict, Optional, Tuple


@attr.s
class DataSplit:
    train: Optional[DataSource] = attr.ib(default=None)
    validate: Optional[DataSource] = attr.ib(default=None)
    predict: Optional[DataSource] = attr.ib(default=None)

    def __getattr__(self, name):
        value = None
        for split in [self.train, self.validate, self.predict]:
            if split is not None:
                new_value = getattr(split, name)
                if value is None and new_value is not None:
                    value = new_value
                elif new_value is not None:
                    assert value == new_value, f"{(value, new_value)}"
        return value


converter.register_unstructure_hook(
    Optional[List[Tuple[DataKey, DataSource]]],
    lambda o: [
        (
            {**converter.unstructure(e[0], unstructure_as=DataKey)},
            {**converter.unstructure(e[1], unstructure_as=DataSource)},
        )
        for e in o
    ]
    if o is not None
    else None,
)


@attr.s
class Dataset:
    name: str = attr.ib()
    id: Optional[str] = attr.ib(default=None)

    train_sources: Optional[List[Tuple[DataKey, DataSource]]] = attr.ib(
        default=attr.Factory(list)
    )
    validate_sources: Optional[List[Tuple[DataKey, DataSource]]] = attr.ib(
        default=attr.Factory(list)
    )
    predict_sources: Optional[List[Tuple[DataKey, DataSource]]] = attr.ib(
        default=attr.Factory(list)
    )

    train_padding: PaddingOption = attr.ib(
        default=PaddingOption.VALID,
    )
    validate_padding: PaddingOption = attr.ib(
        default=PaddingOption.VALID,
    )
    predict_padding: PaddingOption = attr.ib(
        default=PaddingOption.VALID,
    )

    dims: int = attr.ib(default=3)

    def verify(self):
        unstructured = converter.unstructure(self)
        structured = converter.structure(unstructured, self.__class__)
        assert self == structured
        return True

    def __attrs_post_init__(self):
        # add attrs for each array and graph key
        arrays = set(
            [
                key
                for key, _ in itertools.chain(
                    self.train_sources, self.validate_sources, self.predict_sources
                )
                if isinstance(key, ArrayKey)
            ]
        )
        graphs = set(
            [
                key
                for key, _ in itertools.chain(
                    self.train_sources, self.validate_sources, self.predict_sources
                )
                if isinstance(key, GraphKey)
            ]
        )
        for array in arrays:
            self.__setattr__(array.value, DataSplit())
        for graph in graphs:
            self.__setattr__(graph.value, DataSplit())

        for key, source in self.train_sources:
            getattr(self, key.value).train = source

        for key, source in self.validate_sources:
            getattr(self, key.value).validate = source

        for key, source in self.predict_sources:
            getattr(self, key.value).predict = source
