import attr

from typing import Tuple


@attr.s
class TaskConfig:
    """
    Base class for task configurations. Each subclass of a `Task` should
    have a corresponding config class derived from `TaskConfig`.

    Attributes:
        name: A unique name for this task. This will be saved so you and
            others can find and reuse this task. Keep it short and avoid
            special characters.
    Methods:
        verify(self) -> Tuple[bool, str]: This method verifies the TaskConfig object.
    Notes:
        This is a base class for all task configurations. It is not meant to be
        used directly.
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

        Returns:
            Tuple[bool, str]: A tuple containing a boolean value indicating whether the TaskConfig object is valid
                and a string containing the reason why the object is invalid.
        Raises:
            NotImplementedError: This method is not implemented.
        Examples:
            >>> valid, reason = task_config.verify()
        """
        return True, "No validation for this Task"
