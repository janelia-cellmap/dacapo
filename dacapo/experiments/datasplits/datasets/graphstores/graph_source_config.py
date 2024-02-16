import attr


@attr.s
class GraphStoreConfig:
    """Base class for graph store configurations. Each subclass of a
    `GraphStore` should have a corresponding config class derived from
    `GraphStoreConfig`.
    """
