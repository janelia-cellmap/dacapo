from dacapo.experiments.trainers import DummyTrainerConfig

dummy_trainer_config = DummyTrainerConfig(
    name="dummy_trainer", learning_rate=1e-5, batch_size=10, mirror_augment=True
)
