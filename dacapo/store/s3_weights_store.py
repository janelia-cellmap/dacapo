import boto3
import torch
import json
from botocore.exceptions import NoCredentialsError
from dacapo.experiments.run import Run
from .weights_store import WeightsStore, Weights
from typing import Optional


class S3WeightsStore(WeightsStore):
    def __init__(self, s3_path: str):
        """
        Initialize the S3 weights store.

        Args:
            s3_path: The S3 bucket path where the weights are stored.
        """
        if s3_path is None:
            raise ValueError("S3 bucket base path cannot be None")
        self.s3_client = boto3.client("s3")
        self.bucket, self.base_path = self.parse_s3_path(s3_path)

    def parse_s3_path(self, s3_path):
        """Extract bucket and path from the full s3 path."""
        if not s3_path.startswith("s3://"):
            raise ValueError("S3 path must start with 's3://'")
        parts = s3_path[len("s3://") :].split("/", 1)
        return parts[0], parts[1] if len(parts) > 1 else ""

    def store_weights(self, run: Run, iteration: int):
        """
        Store the network weights of the given run on S3.
        """
        weights = Weights(run.model.state_dict(), run.optimizer.state_dict())
        weights_name = f"{self.base_path}/{run.name}/checkpoints/iterations/{iteration}"
        temp_file = f"/tmp/{weights_name.replace('/', '_')}"
        torch.save(weights, temp_file)
        self.s3_client.upload_file(temp_file, self.bucket, weights_name)

    def retrieve_weights(self, run: str, iteration: int) -> Weights:
        """
        Retrieve the network weights of the given run from S3.
        """
        weights_name = f"{self.base_path}/{run}/checkpoints/iterations/{iteration}"
        temp_file = f"/tmp/{weights_name.replace('/', '_')}"
        self.s3_client.download_file(self.bucket, weights_name, temp_file)
        weights = torch.load(temp_file, map_location="cpu")
        return weights

    # Implement other methods like latest_iteration, remove, store_best, retrieve_best etc. using S3 operations.


# Example usage
# s3_store = S3WeightsStore("s3://my-bucket/path/to/weights")
