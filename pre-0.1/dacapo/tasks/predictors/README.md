# Predictor

To add support for a new `Predictor`, you must create an `attr.s` class that
subclasses the `PredictorABC` like this:

```python
from .predictor_abc import PredictorABC

import attr


@attr.s
class MyPredictor(PredictorABC):
    name: str = attr.ib(default="my_predictor")
    fmaps_out: int = attr.ib(default=3)

    param_a: Optional[float] = attr.ib(default=None)
    param_b: Optional[int] = attr.ib(default=None)

    # attributes that can be read from other configurable classes
    fmaps_in: Optional[int] = attr.ib(default=None) # read from model
    dims: Optional[int] = attr.ib(default=None) # read from dataset

    def head(self, fmaps_in: int):
        conv_layer = MyConvLayer(fmaps_in, self.fmaps_out)
        return torch.nn.Sequential(conv_layer, sigmoid)

    def add_target(self, gt, target, weights=None, mask=None):

        target_node = MyTargetNode(*args, **kwargs)
        weights_node = MyWeightsNode(*args, **kwargs)

        return target_node, weights_node
```

`Predictors` require the `head` method, which recieves the `fmaps_in` from `Model`,
and the `add_target` function that should provide a node that generates the targets
from the ground truth. It can also optionally also provide a node to generate
training weights.
`Predictors` must also have a property `fmaps_out`. This is to let DaCapo initialize
zarr datasets into which we can write. This can either be a configurable `attr.ib`
or a `@property` defined on your `Predictor`.

Once you have added support for a new Predictor, you must:
1) import it into `__init__.py`
2) include it in the `AnyPredictor` Union type.
3) Add it to the list of exposed configurable types in `dacapo.configurables`. This Allows
the dacapo-dashboard user interface to read the parameters with their types and metadata.