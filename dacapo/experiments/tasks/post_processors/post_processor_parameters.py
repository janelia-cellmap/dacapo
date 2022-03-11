import attr


@attr.s(frozen=True)
class PostProcessorParameters:
    """Base class for post-processor parameters."""

    id: int = attr.ib()

    @property
    def parameter_names(self):
        return ["id"]
