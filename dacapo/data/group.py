from abc import ABC, abstractmethod


class DatasetGroup:
    """
    A group of datasets. In general, you will have 2 different
    versions of each dataset. i.e. train and validate raw.
    train and validate mask. train and validate gt. etc.

    These are grouped here. It is expected that many attributes
    between train and validate will stay consistent. i.e. axes, num_channels,
    etc. Thus if you try to get an attribute from the group via
    group.num_channels, it simply gets group.train.num_channels.

    If you specifically want an attribute from the validation dataset
    you must explicitly ask for group.valiate.attribute
    """

    def __init__(self):
        self.train = None
        self.validate = None

    def __getattr__(self, attr):
        if attr == "train":
            return self.train
        elif attr == "validate":
            return self.validate
        else:
            return getattr(self.train, attr)