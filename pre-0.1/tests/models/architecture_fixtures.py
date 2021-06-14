from dacapo.models import UNet, VGGNet

unet = UNet(
    # standard model attributes
    input_shape=[20, 20],
    fmaps_out=5,
    # unet attributes
    fmap_inc_factor=2,
    downsample_factors=[[2, 2]],
)
vggnet = VGGNet(
    input_shape=[20, 20],
    output_shape=[1, 1],
    fmaps_out=3,
)

ARCHITECTURES = [unet, vggnet]