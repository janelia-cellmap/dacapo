import attr

from typing import List


@attr.s(frozen=True)
class PostProcessorParameters:
    id: int = attr.ib()

    @property
    def parameter_names(self) -> List[str]:
        return ["id"]


# TODO: Add parameter_names to subclasses
