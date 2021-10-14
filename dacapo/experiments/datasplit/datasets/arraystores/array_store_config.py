import attr


@attr.s
class ArrayStoreConfig:
    """Base class for array source configurations. Each subclass of an
    `ArraySource` should have a corresponding config class derived from
    `ArraySourceConfig`.
    """

    name: str = attr.ib(
        metadata={
            "help_text": "A unique name for this array store. This will be saved so you "
            "and others can find and reuse this array store. Keep it short "
            "and avoid special characters."
        }
    )
