import attr


@attr.s
class PostProcessorParameters:
    """Base class for post-processor parameters."""

    id: int = attr.ib()
