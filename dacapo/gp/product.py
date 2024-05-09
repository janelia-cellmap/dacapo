import gunpowder as gp


class Product(gp.BatchFilter):
    """
    A BatchFilter that multiplies two input arrays and produces an output array.

    Attributes:
        x1_key (:class:`ArrayKey`): The key of the first input array.
        x2_key (:class:`ArrayKey`): The key of the second input array.
        y_key (:class:`ArrayKey`): The key of the output array.
    Provides:
        y_key (gp.ArrayKey): The key of the output array.
    Method:
        __init__: Initialize the Product BatchFilter.
        setup: Set up the Product BatchFilter.
        prepare: Prepare the Product BatchFilter.
        process: Process the Product BatchFilter.

    """

    def __init__(self, x1_key: gp.ArrayKey, x2_key: gp.ArrayKey, y_key: gp.ArrayKey):
        """
        Initialize the Product BatchFilter.

        Args:
            x1_key (gp.ArrayKey): The key of the first input array.
            x2_key (gp.ArrayKey): The key of the second input array.
            y_key (gp.ArrayKey): The key of the output array.
        Raises:
            AssertionError: If the input arrays are not provided.
        Examples:
            >>> Product(x1_key, x2_key, y_key)
            Product(x1_key, x2_key, y_key)
        """
        self.x1_key = x1_key
        self.x2_key = x2_key
        self.y_key = y_key

    def setup(self):
        """
        Set up the Product BatchFilter.

        Enables autoskip and specifies the output array.

        Raises:
            AssertionError: If the input arrays are not provided.
        Examples:
            >>> setup()
            setup()
        """
        self.enable_autoskip()
        self.provides(self.y_key, self.spec[self.x1_key].copy())

    def prepare(self, request):
        """
        Prepare the Product BatchFilter.

        Args:
            request (gp.BatchRequest): The batch request.
        Returns:
            gp.BatchRequest: The dependencies.
        Raises:
            AssertionError: If the input arrays are not provided.
        Examples:
            >>> prepare(request)
            prepare(request)

        """
        deps = gp.BatchRequest()
        deps[self.x1_key] = request[self.y_key].copy()
        deps[self.x2_key] = request[self.y_key].copy()
        return deps

    def process(self, batch, request):
        """
        Process the Product BatchFilter.

        Args:
            batch (gp.Batch): The input batch.
            request (gp.BatchRequest): The batch request.
        Returns:
            gp.Batch: The output batch.
        Raises:
            AssertionError: If the input arrays are not provided.
        Examples:
            >>> process(batch, request)
            process(batch, request)

        """
        outputs = gp.Batch()

        outputs[self.y_key] = gp.Array(
            batch[self.x1_key].data * batch[self.x2_key].data,
            batch[self.x1_key].spec.copy(),
        )

        return outputs
