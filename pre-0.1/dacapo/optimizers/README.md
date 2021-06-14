# Algorithm

An optimizer is just a name with an Algorithm. To add support for a new `Algorithm`, 
you must create an `attr.s` class that subclasses the `AlgorithmABC` like this:

```python
from .algorithm_abc import Algorithm

from typing import Tuple
import attr


@attr.s
class MyAlgorithm(Algorithm):
    my_param1: float = attr.ib(default=0.01)

    def instance(self, params):
        return MyTorchOptimizer(
            params,
            my_param1=self.my_param1
        )
```

`Algorithms` require the `instance` method, which takes in a set of parameters
to optimize, and returns a torch optimizer that can iteratively upgrade the parameters
based on their gradients.

Once you have added support for a new `Algorithm`, you must:
1) import it into `__init__.py`
2) include it in the `AnyAlgorithm` Union type.
3) Add it to the list of exposed configurable types in `dacapo.configurables`. This Allows
the dacapo-dashboard user interface to read the parameters with their types and metadata.