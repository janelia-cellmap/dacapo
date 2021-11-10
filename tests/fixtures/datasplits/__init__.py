from .dummy_datasplit import mk_dummy_datasplit
from .trainable_datasplit import mk_trainable_datasplit

DATASPLIT_MK_FUNCTIONS = [mk_dummy_datasplit, mk_trainable_datasplit]
