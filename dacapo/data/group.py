from abc import ABC, abstractmethod


class DatasetGroup:
    """
    A group of datasets that share some properties
    """

    def __init__(self, datasets=None, properties=None):
        self.datasets = []
        if datasets is not None:
            for dataset in datasets:
                self.datasets.append(dataset)
        self.properties = []
        if properties is not None:
            for prop in properties:
                self.properties.append(prop)

        for prop in self.properties:
            self.__setattr__(prop, self.get_prop(prop))

    def add_dataset(self, dataset):
        self.datasets.append(dataset)

        for prop in self.properties:
            self.assert_all_same(prop)
            self.__setattr__(prop, self.get_prop(prop))

    def assert_all_same(self, prop):
        values = []
        for dataset in self.datasets:
            values.append(getattr(dataset, prop))
        if len(values) <= 1:
            pass
        else:
            first = values[0]
            for i, value in enumerate(values[1:]):
                assert first == value, (
                    "Datasets {self.datasets[0].id} and {self.datasets[i].id} "
                    f"have different values for property {prop} ({first}, {value})"
                )

    def get_prop(self, prop):
        if len(self.datasets) > 0:
            print(F"GETTING ATTRIBUTE {prop} FROM DATASET {self.datasets[0]}")
            v = getattr(self.datasets[0], prop)
            print(F"GOT {v}")

            return v
        else:
            return None
