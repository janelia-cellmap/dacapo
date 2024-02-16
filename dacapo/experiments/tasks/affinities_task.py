class AffinitiesTask(Task):
    """
    This is a class which is a sub-class of Task. It doesn't do any processing logic.
    It is only for definition of the four components: predictor, loss, post_processing,
    evaluator. This class is used in config file to create a series of tasks.

    Attributes:
        predictor: An AffinitiesPredictor object. It is created based on the neighborhood,
                   lsds, affs_weight_clipmin, affs_weight_clipmax, lsd_weight_clipmin,
                   lsd_weight_clipmax, and background_as_object parameters from the input 
                   task config.
        loss: An AffinitiesLoss object. It is created based on the length of neighborhood 
              and lsds_to_affs_weight_ratio parameter from the input task config.
        post_processor: A WatershedPostProcessor object. It is created based on the
                        neighborhood parameter from the input task config. 
        evaluator: An InstanceEvaluator object. It doesn't take parameters during
                   instantiation.
    """

    def __init__(self, task_config):
        """
        This method is for the instantiation of the AffinitiesTask class. It initializes 
        the predictor, loss, post_processor, and evaluator of this class.

        Args:
            task_config (TaskConfig): It is a configuration dictionary containing parameters 
                                      for AffinitiesTask instantiation.

        Returns:
            None.
        """

        self.predictor = AffinitiesPredictor(
            neighborhood=task_config.neighborhood,
            lsds=task_config.lsds,
            affs_weight_clipmin=task_config.affs_weight_clipmin,
            affs_weight_clipmax=task_config.affs_weight_clipmax,
            lsd_weight_clipmin=task_config.lsd_weight_clipmin,
            lsd_weight_clipmax=task_config.lsd_weight_clipmax,
            background_as_object=task_config.background_as_object,
        )
        self.loss = AffinitiesLoss(
            len(task_config.neighborhood), task_config.lsds_to_affs_weight_ratio
        )
        self.post_processor = WatershedPostProcessor(offsets=task_config.neighborhood)
        self.evaluator = InstanceEvaluator()