from abc import ABC
import logging

logger = logging.getLogger(__file__)

        # self.old_head   =["ecs","plasma_membrane","mito","mito_membrane","vesicle","vesicle_membrane","mvb","mvb_membrane","er","er_membrane","eres","nucleus","microtubules","microtubules_out"]
        # self.new_head = ["mito","nucleus","ld","ecs","peroxisome"] 
head_keys = ["prediction_head.weight","prediction_head.bias","chain.1.weight","chain.1.bias"]

# Hack
# if label is mito_peroxisome or peroxisome then change it to mito
mitos = ["mito_proxisome","peroxisome"]

def match_heads(model, head_weights, old_head, new_head ):
    # match the heads
    for label in new_head:
        old_label = label
        if label in mitos:
            old_label = "mito"
        if old_label in old_head:
            logger.warning(f"matching head for {label}")
            # find the index of the label in the old_head
            old_index = old_head.index(old_label)
            # find the index of the label in the new_head
            new_index = new_head.index(label)
            # get the weight and bias of the old head
            for key in head_keys:
                if key in model.state_dict().keys():
                    n_val = head_weights[key][old_index]
                    model.state_dict()[key][new_index] = n_val
            logger.warning(f"matched head for {label} with {old_label}")

class Start(ABC):
    def __init__(self, start_config,remove_head = False, old_head= None, new_head = None):
        self.run = start_config.run
        self.criterion = start_config.criterion
        self.remove_head = remove_head
        self.old_head = old_head
        self.new_head = new_head

    def initialize_weights(self, model):
        from dacapo.store.create_store import create_weights_store

        weights_store = create_weights_store()
        weights = weights_store._retrieve_weights(self.run, self.criterion)

        logger.warning(f"loading weights from run {self.run}, criterion: {self.criterion}")

        try:
            if self.old_head and self.new_head:
                try:
                    self.load_model_using_head_matching(model, weights)
                except RuntimeError as e:
                    logger.error(f"ERROR starter matching head: {e}")
                    self.load_model_using_head_removal(model, weights)
            elif self.remove_head:
                self.load_model_using_head_removal(model, weights)
            else:
                model.load_state_dict(weights.model)
        except RuntimeError as e:
            logger.warning(f"ERROR starter: {e}")

    def load_model_using_head_removal(self, model, weights):
        logger.warning(f"removing head from run {self.run}, criterion: {self.criterion}")
        for key in head_keys:
            weights.model.pop(key, None)
        logger.warning(f"removed head from run {self.run}, criterion: {self.criterion}")
        model.load_state_dict(weights.model, strict=False)
        logger.warning(f"loaded weights in non strict mode from run {self.run}, criterion: {self.criterion}")

    def load_model_using_head_matching(self, model, weights):
        logger.warning(f"matching heads from run {self.run}, criterion: {self.criterion}")
        logger.warning(f"old head: {self.old_head}")
        logger.warning(f"new head: {self.new_head}")
        head_weights = {}
        for key in head_keys:
            head_weights[key] = weights.model[key]
        for key in head_keys:
            weights.model.pop(key, None)
        model.load_state_dict(weights.model, strict=False)
        model = match_heads(model, head_weights, self.old_head, self.new_head)
            


