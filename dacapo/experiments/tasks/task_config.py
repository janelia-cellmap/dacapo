import attr

from typing import Tuple


@attr.s
class TaskConfig:
    """Base class for task configurations. Each subclass of a `Task` should
    have a corresponding config class derived from `TaskConfig`.
    """

    name: str = attr.ib(
        metadata={
            "help_text": "A unique name for this task. This will be saved so you and "
            "others can find and reuse this task. Keep it short and avoid "
            "special characters."
        }
    )

    def verify(self) -> Tuple[bool, str]:
        """
        Check whether this is a valid Task
        """
        return True, "No validation for this Task"
