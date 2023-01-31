import gunpowder as gp

import copy


class GraphSource(gp.BatchProvider):
    def __init__(self, key: gp.GraphKey, graph: gp.Graph):
        self.key = key
        self.graph = graph

    def setup(self):
        self.provides(self.key, self.graph.spec)

    def provide(self, request):
        outputs = gp.Batch()
        outputs[self.key] = copy.deepcopy(
            self.graph.crop(request[self.key].roi).trim(request[self.key].roi)
        )
        return outputs
