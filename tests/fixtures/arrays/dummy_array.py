from dacapo.experiments.datasplits.datasets.arrays import DummyArrayConfig

dummy_array_config = DummyArrayConfig(name="dummy_array")


def mk_dummy_array(*args, **kwargs):
    return dummy_array_config
