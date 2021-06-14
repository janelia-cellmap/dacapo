import gunpowder as gp
import networkx as nx


class NXSource(gp.BatchProvider):
    def __init__(self, out_graph, filename, location_attr="location"):
        self.out_graph = out_graph
        self.filename = filename
        self.location_attr = location_attr

    def setup(self):
        self.graph = gp.Graph.from_nx_graph(nx.read_gpickle(self.filename))
        self.provides(self.out_graph, gp.GraphSpec())

    def provide(self, request):
        output = gp.Batch()
        output[self.out_graph] = self.graph.crop(request[self.out_graph].roi)
        return output
