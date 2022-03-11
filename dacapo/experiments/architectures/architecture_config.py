import attr


@attr.s
class ArchitectureConfig:
    """Base class for architecture configurations. Each subclass of an
    `Architecture` should have a corresponding config class derived from
    `ArchitectureConfig`.
    """

    name: str = attr.ib(
        metadata={
            "help_text": "A unique name for this architecture. This will be saved so "
            "you and others can find and reuse this task. Keep it short "
            "and avoid special characters."
        }
    )
