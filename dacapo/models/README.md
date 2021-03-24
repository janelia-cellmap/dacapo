# Model

A model is just a name with an Architecture. To add support for a new `Architecture`, 
you must create an `attr.s` class that subclasses the `ArchitectureABC` like this:

```python
import attr

from .architecture_abc import ArchitectureABC

from typing import List, Optional


@attr.s
class MyArchitecture(ArchitectureABC):
    # standard model attributes
    input_shape: List[int] = attr.ib()
    output_shape: Optional[List[int]] = attr.ib()
    fmaps_out: int = attr.ib()

    predict_input_shape: Optional[List[int]] = attr.ib()
    predict_output_shape: Optional[List[int]] = attr.ib()

    def instantiate(self, fmaps_in: int):
        return MyTorchModule(fmaps_in, *args, **kwargs)
```

`Architecture`s require the `instantiate` method, which takes `fmaps_in`
and returns a `torch` module that can take a volume of size `input_shape`
with `fmaps_in` channels and return a volume of size `output_shape` with
`fmaps_out` channels.
`Architecture`s must have the `input_shape`, `output_shape`, and `fmaps_out`
properties. These will be used for determining batch sizes during training.
`Architecture`s may also provide the `predict_input_shape` and `predict_output_shape`
arguments to make prediction more efficient. Often times you can get significantly
larger chunks through your model during prediction since you don't have
the memory constraints of keeping gradients and a large batch in memory.


Once you have added support for a new `Architecture`, you must:
1) import it into `__init__.py`
2) include it in the `AnyArchitecture` Union type.
3) Add it to the list of exposed configurable types in `dacapo.configurables`. This Allows
the dacapo-dashboard user interface to read the parameters with their types and metadata.