import gunpowder as gp


class Product(gp.BatchFilter):
    def __init__(self, x1_key: gp.ArrayKey, x2_key: gp.ArrayKey, y_key: gp.ArrayKey):
        self.x1_key = x1_key
        self.x2_key = x2_key
        self.y_key = y_key

    def setup(self):
        self.enable_autoskip()
        self.provides(self.y_key, self.spec[self.x1_key].copy())

    def prepare(self, request):
        deps = gp.BatchRequest()
        deps[self.x1_key] = request[self.y_key].copy()
        deps[self.x2_key] = request[self.y_key].copy()
        return deps

    def process(self, batch, request):
        outputs = gp.Batch()

        outputs[self.y_key] = gp.Array(
            batch[self.x1_key].data * batch[self.x2_key].data,
            batch[self.x1_key].spec.copy(),
        )

        return outputs
