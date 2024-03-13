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
        """
        self.array_key = array_key
        self.copy_key = copy_key
        self.drop_channels = drop_channels

    def setup(self):
        """
        Sets up the filter by enabling autoskip and providing the copied key.
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
