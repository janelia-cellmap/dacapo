import gunpowder as gp
import numpy as np


class Binarize(gp.BatchFilter):
    def __init__(self, input_array, output_array, target=0):
        self.input_array = input_array
        self.output_array = output_array
        self.target = target

    def setup(self):
        spec = self.spec[self.input_array].copy()
        spec.dtype = np.bool
        self.provides(self.output_array, spec)

    def prepare(self, request):
        deps = gp.BatchRequest()
        deps[self.in_array] = request[self.output_array].spec.copy()
        return deps

    def process(self, batch, request):
        outputs = gp.Batch()

        if self.in_array not in batch:
            return

        data = batch[self.input_array].data
        spec = batch[self.input_array].spec.copy()
        spec.dtype = np.bool
        binarized = data == self.target
        outputs[self.output_array] = gp.Array(binarized, spec)

        return outputs
