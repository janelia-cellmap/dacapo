import gunpowder as gp


class CopyMask(gp.BatchFilter):
    """
    A class to copy a mask into a new key with the option to drop channels via max collapse.

    Attributes:
        array_key (gp.ArrayKey): Original key of the array from where the mask will be copied.
        copy_key (gp.ArrayKey): New key where the copied mask will reside.
        drop_channels (bool): If True, channels will be dropped via a max collapse.
    Methods:
        setup: Sets up the filter by enabling autoskip and providing the copied key.
        prepare: Prepares the filter by copying the request of copy_key into a dependency.
        process: Processes the batch by copying the mask from the array_key to the copy_key.
    Note:
        This class is a subclass of gunpowder.BatchFilter and is used to
        copy a mask into a new key with the option to drop channels via max collapse.
    """

    def __init__(
        self, array_key: gp.ArrayKey, copy_key: gp.ArrayKey, drop_channels: bool = False
    ):
        """
        Constructs the necessary attributes for the CopyMask object.

        Args:
            array_key (gp.ArrayKey): Original key of the array from where the mask will be copied.
            copy_key (gp.ArrayKey): New key where the copied mask will reside.
            drop_channels (bool): If True, channels will be dropped via a max collapse. Default is False.
        Raises:
            TypeError: If array_key is not of type gp.ArrayKey.
            TypeError: If copy_key is not of type gp.ArrayKey.
        Examples:
            >>> array_key = gp.ArrayKey("ARRAY")
            >>> copy_key = gp.ArrayKey("COPY")
            >>> copy_mask = CopyMask(array_key, copy_key)
        """
        self.array_key = array_key
        self.copy_key = copy_key
        self.drop_channels = drop_channels

    def setup(self):
        """
        Sets up the filter by enabling autoskip and providing the copied key.

        Raises:
            RuntimeError: If the key is already provided.
        Examples:
            >>> copy_mask.setup()

        """
        self.enable_autoskip()
        self.provides(self.copy_key, self.spec[self.array_key].copy())

    def prepare(self, request):
        """
        Prepares the filter by copying the request of copy_key into a dependency.

        Args:
            request: The request to prepare.
        Returns:
            deps: The prepared dependencies.
        Raises:
            NotImplementedError: If the copy_key is not provided.
        Examples:
            >>> request = gp.BatchRequest()
            >>> request[self.copy_key] = gp.ArraySpec(roi=gp.Roi((0, 0, 0), (1, 1, 1)))
            >>> copy_mask.prepare(request)
        """
        deps = gp.BatchRequest()
        deps[self.array_key] = request[self.copy_key].copy()
        return deps

    def process(self, batch, request):
        """
        Processes the batch by copying the mask from the array_key to the copy_key.

        If "drop_channels" attribute is True, it performs max collapse.

        Args:
            batch: The batch to process.
            request: The request for processing.
        Returns:
            outputs: The processed outputs.
        Raises:
            KeyError: If the requested key is not in the request.
        Examples:
            >>> request = gp.BatchRequest()
            >>> request[gp.ArrayKey("ARRAY")] = gp.ArraySpec(roi=gp.Roi((0, 0, 0), (1, 1, 1)))
            >>> copy_mask.process(batch, request)
        """
        outputs = gp.Batch()

        outputs[self.copy_key] = batch[self.array_key]
        if self.drop_channels:
            while (
                outputs[self.copy_key].data.ndim
                > outputs[self.copy_key].spec.voxel_size.dims
            ):
                outputs[self.copy_key].data = outputs[self.copy_key].data.max(axis=0)

        return outputs
