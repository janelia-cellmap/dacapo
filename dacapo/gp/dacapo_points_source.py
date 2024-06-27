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
    Methods:
        setup(): Set up the provider.
        provide(request): Provides the graph for the requested ROI.

    Note:
        This class is a subclass of gunpowder.BatchProvider and is used to
        serve graph data to gunpowder pipelines.
    """

    def __init__(self, key: gp.GraphKey, graph: gp.Graph):
        """
        Args:
            key (gp.GraphKey): The key of the graph to be served.
            graph (gp.Graph): The graph to be served.
        Raises:
            TypeError: If key is not of type gp.GraphKey.
            TypeError: If graph is not of type gp.Graph.
        Examples:
            >>> graph = gp.Graph()
            >>> graph.add_node(1, position=[0, 0, 0])
            >>> graph.add_node(2, position=[1, 1, 1])
            >>> graph.add_edge(1, 2)
            >>> graph_source = GraphSource(gp.GraphKey("GRAPH"), graph)
        """
        self.key = key
        self.graph = graph

    def setup(self):
        """
        Set up the provider. This function sets the provider to provide the
        graph with the given key.

        Raises:
            RuntimeError: If the key is already provided.
        Examples:
            >>> graph_source.setup()

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
        Raises:
            KeyError: If the requested key is not in the request.
        Examples:
            >>> request = gp.BatchRequest()
            >>> request[gp.GraphKey("GRAPH")] = gp.GraphSpec(roi=gp.Roi((0, 0, 0), (1, 1, 1)))
            >>> graph_source.provide(request)
        """
        outputs = gp.Batch()
        if self.key in request:
            outputs[self.key] = copy.deepcopy(
                self.graph.crop(request[self.key].roi).trim(request[self.key].roi)
            )
        return outputs
