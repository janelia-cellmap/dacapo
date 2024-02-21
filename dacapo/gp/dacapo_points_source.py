import gunpowder as gp

import copy


class GraphSource(gp.BatchProvider):
    """
    A provider for serving graph data in gunpowder pipelines.

    The Graph Source loads a single graph to serve to the pipeline based on
    ROI requests it receives.

    Attributes:
        key (gp.GraphKey): The key of the graph to be served.
        graph (gp.Graph): The graph to be served.
    """

    def __init__(self, key: gp.GraphKey, graph: gp.Graph):
        """
        Args:
            key (gp.GraphKey): The key of the graph to be served.
            graph (gp.Graph): The graph to be served.
        """
        self.key = key
        self.graph = graph

    def setup(self):
        """
        Set up the provider. This function sets the provider to provide the
        graph with the given key.
        """
        self.provides(self.key, self.graph.spec)

    def provide(self, request):
        """
        Provides the graph for the requested ROI.

        This method will be passively called by gunpowder to get a batch.
        Depending on the request we provide a subgraph of our data, or nothing
        at all.

        Args:
            request (gp.BatchRequest): BatchRequest with the same ROI for
            each requested array and graph.

        Returns:
            outputs (gp.Batch): The graph contained in a Batch.
        """
        outputs = gp.Batch()
        if self.key in request:
            outputs[self.key] = copy.deepcopy(
                self.graph.crop(request[self.key].roi).trim(request[self.key].roi)
            )
        return outputs
