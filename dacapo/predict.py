```python
def predict(
    run_name: str,
    iteration: int,
    input_container: Path | str,
    input_dataset: str,
    output_path: Path | str,
    output_roi: Optional[str] = None,
    num_workers: int = 30,
    output_dtype: np.dtype | str = np.uint8,  # type: ignore
    compute_context: ComputeContext | str = LocalTorch(),
    overwrite: bool = True,
):
    """
    Method to perform prediction using a specified model iteration on a given input dataset. The result is
    dumped in a specified output path. Region of interest(roi) to predict on can also be specified while running prediction.
    In case roi is not provided, it's set to the raw roi. The prediction is performed in a parallelized manner using
    the given number of workers.

    Args:
        run_name (str): Name of the run to be used for prediction.
        iteration (int): The iteration of the model to be used for prediction.
        input_container (Path or str): Container contains the raw data to be predicted.
        input_dataset (str): The dataset to be used for prediction.
        output_path (Path or str): The path where prediction results are written.
        output_roi (str): Region of interest to perform prediction on.If not given, raw roi will be used.
        num_workers (int): Number of workers used to perform prediction in parallel. Defaults is 30.
        output_dtype (np.dtype or str): The dtype of the prediction output. Defaults to np.uint8.
        compute_context (ComputeContext or str): Computation context to use for prediction. Must be the name of a subclass of ComputeContext.
        Defaults to LocalTorch(), which means the prediction runs on the local machine without any special hardware acceleration.
        overwrite (bool, optional): Flag to allow overwriting existent prediction file stored in output_path. If False, prediction will not overwrite. Defaults to True.

    Returns:
        None
    """
```