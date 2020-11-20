import gunpowder as gp
import numpy as np


class BinarizeNot(gp.BatchFilter):
    def __init__(self, in_array, out_array, target=0):
        self.in_array = in_array
        self.out_array = out_array
        self.target = target

    def setup(self):
        spec = self.spec[self.in_array].copy()
        spec.dtype = np.bool
        self.provides(self.out_array, spec)

    def prepare(self, request):
        deps = gp.BatchRequest()
        spec = gp.ArraySpec(roi=request[self.out_array].roi)
        deps[self.in_array] = spec
        return deps

    def process(self, batch, request):
        outputs = gp.Batch()

        if self.in_array not in batch:
            return

        data = batch[self.in_array].data
        spec = batch[self.in_array].spec.copy()
        spec.dtype = np.bool
        binarized = data != self.target
        outputs[self.out_array] = gp.Array(binarized, spec)

        return outputs
