from dacapo.experiments.datasplits import DummyDataSplitConfig

dummy_datasplit_config = DummyDataSplitConfig(name="dummy_datasplit")


def mk_dummy_datasplit(*args, **kwargs):
    return dummy_datasplit_config
