import attr


@attr.s
class TaskConfig:
    """Base class for task configurations. Each subclass of a `Task` should
    have a corresponding config class derived from `TaskConfig`.
    """

    name: str = attr.ib(
        metadata={
            "help_text":
                "A unique name for this task. This will be saved so you and "
                "others can find and reuse this task. Keep it short and avoid "
                "special characters."
        }
    )
