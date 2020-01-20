from .shallow_learner_chain import ShallowLearnerChain
from .unet import UNet


def create_model(model_config, fmaps_in, fmaps_out):

    if model_config.type == ShallowLearnerChain:

        base_model_config = model_config.base_model

        def create_shallow_learner(fmaps_in, fmaps_out):
            return create_model(base_model_config, fmaps_in, fmaps_out)

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
