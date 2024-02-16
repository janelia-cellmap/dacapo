import pytorch_lightning as pl
from omegaconf import DictConfig
from dacapo.task_wrappers import PretrainedTaskConfig


class Dacapo(pl.LightningModule):
    """
    A PyTorch Lightning Module for the Dacapo Python library.

    This module is used to combine different tasks or algorithms which will be run consecutively.
    It also allows starting any task with pretrained weights.

    Attributes:
        task (PretrainedTaskConfig): The configuration for the sub-task to run starting with
            the provided pretrained weights.
    """

    def __init__(self, task):
        super().__init__()
        self.task = task

    def forward(self, x):
        """
        Forward propagation function. It runs the set of tasks on the input data sequentially.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            The output of the final task in the sequence.
        """
        return self.task(x)

    def training_step(self, batch, batch_idx):
        """
        Executes a single training step. This computes the loss for the current task.

        Args:
            batch (torch.Tensor): The current batch of data for training.
            batch_idx (int): The index of the current batch.

        Returns:
            A dictionary containing the loss to backpropagate.
        """
        x, y = batch
        y_hat = self.task(x)
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss)
        return {'loss': loss}

    @staticmethod
    def from_config(config: DictConfig):
        """
        Create Dacapo instance from a given config.

        Args:
            config (DictConfig): A configuration object to initialize the Dacapo instance.

        Returns:
            A new Dacapo instance with the specified settings.
        """
        task = PretrainedTaskConfig.from_config(config.task)
        return Dacapo(task)
