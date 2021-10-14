import attr


@attr.s
class DataSplitConfig:
    """Base class for datasplit configurations. Each subclass of an
    `DataSplit` should have a corresponding config class derived from
    `DataSplitConfig`.
    """

    name: str = attr.ib(
        metadata={
            "help_text":
                "A unique name for this datasplit. This will be saved so "
                "you and others can find and reuse this datasplit. Keep it short "
                "and avoid special characters."
        }
    )
