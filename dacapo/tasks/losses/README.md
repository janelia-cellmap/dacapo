# Losses

To add support for a new Loss, you must create an `attr.s` class that
subclasses the `LossABC` like this:

```python
from .loss_abc import LossABC

import attr

from typing import Optional


@attr.s
class MyLoss(LossABC):
    param_a: float = attr.ib()

    def instantiate(self):
        return MyTorchLoss(param_a)
```

The only required function is `instantiate`, that returns a torch Module. The returned
module will then recieve the predictions, targets, and optional weights with which
to compute the loss.

Once you have added support for a new loss, you must:
1) import it into `__init__.py`
2) include it in the `AnyLoss` Union type.
3) Add it to the list of exposed configurable types in `dacapo.configurables`. This Allows
the dacapo-dashboard user interface to read the parameters with their types and metadata.