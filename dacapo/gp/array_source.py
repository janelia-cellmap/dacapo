import gunpowder as gp


class ArraySource(gp.BatchProvider):
    def __init__(self, data, key, spec):
        self.array = gp.Array(data, spec)
        self.key = key
        self.array_spec = spec

    def setup(self):
        self.provides(self.key, self.array_spec.copy())

    def provide(self, request):
        output = gp.Batch()

        output[self.key] = self.array.crop(request[self.key].roi)

        return output
