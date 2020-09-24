from abc import ABC, abstractmethod


class DatasetGroup:
    """
    A group of datasets that share some properties
    """

    def __init__(self):
        self.train = None
        self.validate = None

