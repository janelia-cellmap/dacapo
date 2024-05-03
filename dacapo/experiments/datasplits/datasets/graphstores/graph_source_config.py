import attr


@attr.s
class GraphStoreConfig:
    """
    Base class for graph store configurations. Each subclass of a
    `GraphStore` should have a corresponding config class derived from
    `GraphStoreConfig`.

    Attributes:
        store_type (class): The type of graph store that is being configured.
    Methods:
        verify: A method to verify the validity of the configuration.
    Notes:
        This class is used to create a configuration object for the graph store.
    """

    pass
