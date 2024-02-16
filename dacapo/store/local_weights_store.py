```python
class LocalWeightsStore(WeightsStore):
	"""
	A local store for network weights providing various methods to manage (store, retrieve, remove) weights.
	
	Methods
	-------
	__init__(self, basedir):
		Initializes a local weights store at the given directory base directory.

	latest_iteration(self, run: str) -> Optional[int]:
		Returns the latest iteration for which weights are available for the given run.

	store_weights(self, run: Run, iteration: int):
		Stores the network weights of the provided run for the given iteration.

	retrieve_weights(self, run: str, iteration: int) -> Weights:
		Retrieves the network weights of the given run for the given iteration.

	_retrieve_weights(self, run: str, key: str) -> Weights:
		Retrieves weights using the provided run and key.

	remove(self, run: str, iteration: int):
		Removes weights associated with the provided run and iteration.

	store_best(self, run: str, iteration: int, dataset: str, criterion: str):
		Stores the best weights in an easily findable location based on the given run, iteration, dataset, and criterion.

	retrieve_best(self, run: str, dataset: str | Dataset, criterion: str) -> int:
		Retrieves the best iteration from the given run, dataset and criterion.

	_load_best(self, run: Run, criterion: str):
		Retrieves the weights for the given run and criterion, and loads it into the model.

	__get_weights_dir(self, run: Union[str, Run]):
		Returns the weight directory path for the provided run.
	"""
```