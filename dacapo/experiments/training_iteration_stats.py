```python
import attr

@attr.s
class TrainingIterationStats:
    """
    A class to represent the training iteration statistics.

    Attributes:
        iteration (int): The iteration that produced these stats.
        loss (float): The loss value of this iteration.
        time (float): The time it took to process this iteration.

    """
    iteration: int = attr.ib(
        metadata={"help_text": "The iteration that produced these stats."}
    )
    loss: float = attr.ib(metadata={"help_text": "The loss of this iteration."})
    time: float = attr.ib(
        metadata={"help_text": "The time it took to process this iteration."}
    )
```
