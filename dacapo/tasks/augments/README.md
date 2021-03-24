# Augmentations

To add support for a new augmentation, you must create an `attr.s` class that
subclasses the `AugmetABC` like this:

```python
from .augment_abc import AugmentABC

import attr


@attr.s
class MyAugment(AugmentABC):
    param_a: int = attr.ib(metadata={"help_text": "an integer parameter"})
    param_b: float = attr.ib(metadata={"help_text": "a float parameter"})
    param_c: str = attr.ib(metadata={"help_text": "a string parameter"})
    param_d: bool = attr.ib(metadata={"help_text": "a bool parameter"})

    def node(self, array):
        return MyGunpowderAugmentNode(self.param_a, self.param_b, self.param_c, self.param_d)
```

The only required function is `node`, that returns a gunpowder node that
applies your desired augmentation. `node` will always recieve an arraykey
for raw which you may use (like IntensityAugment), or simply operate on everything that is
requested (like SimpleAugment or ElasticAugment).

Once you have added support for a new augmentation, you must:
1) import it into `__init__.py`
2) include it in the `AnyAugment` Union type.
3) Add it to the list of exposed configurable types in `dacapo.configurables`. This Allows
the dacapo-dashboard user interface to read the parameters with their types and metadata.