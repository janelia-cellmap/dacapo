The docstrings for the apply and apply_run functions could be written as follows:

```python
def apply(
    run_name: str,
    input_container: Path | str,
    input_dataset: str,
    output_path: Path | str,
    validation_dataset: Optional[Dataset | str] = None,
    criterion: str = "voi",
    iteration: Optional[int] = None,
    parameters: Optional[PostProcessorParameters | str] = None,
    roi: Optional[Roi | str] = None,
    num_cpu_workers: int = 30,
    output_dtype: Optional[np.dtype | str] = np.uint8,  # type: ignore
    compute_context: ComputeContext = LocalTorch(),
    overwrite: bool = True,
    file_format: str = "zarr",
):
    """
    Loads weights and applies a model to a given dataset.

    Args:
        run_name (str): The name of the run.
        input_container (Path|str): Input dataset path.
        input_dataset (str): The input dataset.
        output_path (Path|str): The output directory path.
        validation_dataset(Optional[Dataset|str], optional): Dataset for validation. Defaults to None.
        criterion (str, optional): The criterion to be used. Defaults to "voi".
        iteration (Optional[int], optional): The iteration number. If None, uses the best iteration based on the criterion. Defaults to None.
        parameters (Optional[PostProcessorParameters|str], optional): Model parameters. If None, uses the best parameters for the validation dataset. Defaults to None.
        roi (Optional[Roi|str], optional): The region of interest. If None, the whole input dataset is used. Defaults to None.
        num_cpu_workers (int, optional): Number of workers for the CPU. Defaults to 30.
        output_dtype(Optional[np.dtype|str], optional): The datatype for the output. Defaults to np.uint8.
        compute_context (ComputeContext, optional): The computation context. Defaults to LocalTorch().
        overwrite (bool, optional): Whether to overwrite existing files or not. Defaults to True.
        file_format (str, optional): The file format for output files. Defaults to "zarr".

    Raises:
        ValueError: If validation_dataset is not provided as required.
        ValueError: If provided parameters string is not parsable.
        Exception: If unable to instantiate post-processor with given arguments.
    """
...
def apply_run(
    run: Run,
    parameters: PostProcessorParameters,
    input_array: Array,
    prediction_array_identifier: "LocalArrayIdentifier",
    output_array_identifier: "LocalArrayIdentifier",
    roi: Optional[Roi] = None,
    num_cpu_workers: int = 30,
    output_dtype: Optional[np.dtype] = np.uint8,  # type: ignore
    compute_context: ComputeContext = LocalTorch(),
    overwrite: bool = True,
):
    """Apply the model to a given dataset. Assumes model is already loaded.

    Args:
        run (Run): The runtime object.
        parameters (PostProcessorParameters): Model parameters.
        input_array (Array): The input array to the model.
        prediction_array_identifier ("LocalArrayIdentifier"): Identifier for the prediction array.
        output_array_identifier ("LocalArrayIdentifier"): Identifier for the output array.
        roi (Optional[Roi], optional): The region of interest. If None, the whole input dataset is used. Defaults to None.
        num_cpu_workers (int, optional): Number of workers for the CPU. Defaults to 30.
        output_dtype (Optional[np.dtype], optional): Datatype for the output. Defaults to np.uint8.
        compute_context (ComputeContext, optional): The computation context. Defaults to LocalTorch().
        overwrite (bool, optional): Whether to overwrite existing files or not. Defaults to True.
    """
...
```