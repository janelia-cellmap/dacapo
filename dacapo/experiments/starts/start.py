from abc import ABC
import logging

logger = logging.getLogger(__file__)

# self.old_head   =["ecs","plasma_membrane","mito","mito_membrane","vesicle","vesicle_membrane","mvb","mvb_membrane","er","er_membrane","eres","nucleus","microtubules","microtubules_out"]
# self.new_head = ["mito","nucleus","ld","ecs","peroxisome"]


def match_heads(model, weights, old_head, new_head):
    # match the heads
    for label in new_head:
        if label in old_head:
            logger.warning(f"matching head for {label}")
            # find the index of the label in the old_head
            old_index = old_head.index(label)
            # find the index of the label in the new_head
            new_index = new_head.index(label)
            # get the weight and bias of the old head
            for key in [
                "prediction_head.weight",
                "prediction_head.bias",
                "chain.1.weight",
                "chain.1.bias",
            ]:
                if key in model.state_dict().keys():
                    n_val = weights.model[key][old_index]
                    model.state_dict()[key][new_index] = n_val
            logger.warning(f"matched head for {label}")
    return model


class Start(ABC):
    def __init__(self, start_config, remove_head=False, old_head=None, new_head=None):
        self.run = start_config.run
        self.criterion = start_config.criterion
        self.remove_head = remove_head
        self.old_head = old_head
        self.new_head = new_head

    def initialize_weights(self, model):
        from dacapo.store.create_store import create_weights_store

        weights_store = create_weights_store()
        weights = weights_store._retrieve_weights(self.run, self.criterion)

        logger.info(f"loading weights from run {self.run}, criterion: {self.criterion}")

        try:
            if self.old_head and self.new_head:
                logger.warning(
                    f"matching heads from run {self.run}, criterion: {self.criterion}"
                )
                logger.info(f"old head: {self.old_head}")
                logger.info(f"new head: {self.new_head}")
                model = match_heads(model, weights, self.old_head, self.new_head)
                logger.warning(
                    f"matched heads from run {self.run}, criterion: {self.criterion}"
                )
                self.remove_head = True
            if self.remove_head:
                logger.warning(
                    f"removing head from run {self.run}, criterion: {self.criterion}"
                )
                weights.model.pop("prediction_head.weight", None)
                weights.model.pop("prediction_head.bias", None)
                weights.model.pop("chain.1.weight", None)
                weights.model.pop("chain.1.bias", None)
                logger.warning(
                    f"removed head from run {self.run}, criterion: {self.criterion}"
                )
                model.load_state_dict(weights.model, strict=False)
                logger.warning(
                    f"loaded weights in non strict mode from run {self.run}, criterion: {self.criterion}"
                )
            else:
                model.load_state_dict(weights.model)
        except RuntimeError as e:
            logger.warning(e)
