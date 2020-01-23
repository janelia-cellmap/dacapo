from .shallow_learner_chain import ShallowLearnerChain
from .unet import UNet
from again.prediction_types import Affinities


def create_model(task_config, model_config, fmaps_in=None, fmaps_out=None):

    if fmaps_in is None:
        fmaps_in = task_config.data.channels

    if fmaps_out is None:
        fmaps_out = {
            Affinities: task_config.data.dims
        }[task_config.predict]

    if model_config.type == ShallowLearnerChain:

        base_model_config = model_config.base_model

        def create_shallow_learner(fmaps_in, fmaps_out):
            return create_model(
                task_config,
                base_model_config,
                fmaps_in,
                fmaps_out)

        model = ShallowLearnerChain(
            fmaps_in, fmaps_out,
            num_learners=model_config.num_learners,
            num_latents=model_config.num_latents,
            create_shallow_learner=create_shallow_learner)

        return model

    elif model_config.type == UNet:

        model = UNet(
            fmaps_in=fmaps_in,
            fmaps=model_config.fmaps,
            fmaps_out=fmaps_out,
            fmap_inc_factor=model_config.fmap_inc_factor,
            downsample_factors=model_config.downsample_factors,
            padding=model_config.padding)

        return model

    else:

        raise RuntimeError(
            f"create_model does (yet) not support {model_config.type}")
