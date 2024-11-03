import numpy as np
import logging

logger = logging.getLogger(__name__)

try:
    from empanada_napari.inference import Engine3d
    from empanada_napari.multigpu import MultiGPUEngine3d
    from empanada_napari.utils import get_configs
    from empanada.config_loaders import read_yaml
    from empanada_napari.inference import (
        InstanceTracker,
        get_axis_trackers_by_class,
        instance_relabel,
        filters,
        fill_volume,
        create_instance_consensus,
        create_semantic_consensus,
    )
except ImportError:
    raise ImportError("Please install empanada-napari to use this CLI")


default_parameters = {
    "model_config": "MitoNet_v1",
    "use_gpu": True,
    "use_quantized": False,
    "multigpu": False,
    "downsampling": 1,
    "confidence_thr": 0.5,
    "center_confidence_thr": 0.1,
    "min_distance_object_centers": 21,
    "fine_boundaries": True,
    "semantic_only": False,
    "median_slices": 11,
    "min_size": 10000,
    "min_extent": 50,
    "maximum_objects_per_class": 1000000,
    "inference_plane": "xy",
    "orthoplane": True,
    "return_panoptic": False,
    "pixel_vote_thr": 1,
    "allow_one_view": False,
}


def segment_function(input_array, block, **parameters):
    
    vols, class_names = [], []
    for vol, class_name, _ in empanada_segmenter(
        input_array[block.read_roi], **parameters
    ):
        vols.append(vol[None, ...].astype(np.uint64))
        class_names.append(class_name)
    return np.concatenate(vols, axis=0, dtype=np.uint64).squeeze()


# THESE ARE NON-THREAD WORKER VERSIONS OF THE FUNCTIONS, adapted from empanada-napari in by Jeff Rhoades (HHMI Janelia) February 2024

model_configs = get_configs()


def stack_inference(engine, volume, axis_name):
    
    stack, trackers = engine.infer_on_axis(volume, axis_name)
    trackers_dict = {axis_name: trackers}
    return stack, axis_name, trackers_dict


def orthoplane_inference(engine, volume):
    
    trackers_dict = {}
    for axis_name in ["xy", "xz", "yz"]:
        stack, trackers = engine.infer_on_axis(volume, axis_name)
        trackers_dict[axis_name] = trackers

        # report instances per class
        for tracker in trackers:
            class_id = tracker.class_id
            print(
                f"Class {class_id}, axis {axis_name}, has {len(tracker.instances.keys())} instances"
            )

    return trackers_dict


def empanada_segmenter(
    image,
    model_config="MitoNet_v1",
    use_gpu=True,
    use_quantized=False,
    multigpu=False,
    downsampling=1,
    confidence_thr=0.5,
    center_confidence_thr=0.1,
    min_distance_object_centers=21,
    fine_boundaries=True,
    semantic_only=False,
    median_slices=11,
    min_size=10000,
    min_extent=50,
    maximum_objects_per_class=1000000,
    inference_plane="xy",
    orthoplane=True,
    return_panoptic=False,
    pixel_vote_thr=1,
    allow_one_view=False,
):
    
    # load the model config
    model_config = read_yaml(model_configs[model_config])
    min_size = int(min_size)
    min_extent = int(min_extent)
    maximum_objects_per_class = int(maximum_objects_per_class)

    if multigpu:
        engine = MultiGPUEngine3d(
            model_config,
            inference_scale=downsampling,
            median_kernel_size=median_slices,
            nms_kernel=min_distance_object_centers,
            nms_threshold=center_confidence_thr,
            confidence_thr=confidence_thr,
            min_size=min_size,
            min_extent=min_extent,
            fine_boundaries=fine_boundaries,
            label_divisor=maximum_objects_per_class,
            semantic_only=semantic_only,
            save_panoptic=return_panoptic,
        )
    # conditions where model needs to be (re)loaded
    else:
        engine = Engine3d(
            model_config,
            inference_scale=downsampling,
            median_kernel_size=median_slices,
            nms_kernel=min_distance_object_centers,
            nms_threshold=center_confidence_thr,
            confidence_thr=confidence_thr,
            min_size=min_size,
            min_extent=min_extent,
            fine_boundaries=fine_boundaries,
            label_divisor=maximum_objects_per_class,
            use_gpu=use_gpu,
            use_quantized=use_quantized,
            semantic_only=semantic_only,
            save_panoptic=return_panoptic,
        )

    def start_postprocess_worker(*args):
        
        trackers_dict = args[0][2]
        for vol, class_name, tracker in stack_postprocessing(
            trackers_dict,
            model_config,
            label_divisor=maximum_objects_per_class,
            min_size=min_size,
            min_extent=min_extent,
            dtype=engine.dtype,
        ):
            print(f"Yielding {class_name} volume of shape {vol.shape}")
            yield vol, class_name, tracker

    def start_consensus_worker(trackers_dict):
        
        for vol, class_name, tracker in tracker_consensus(
            trackers_dict,
            model_config,
            pixel_vote_thr=pixel_vote_thr,
            allow_one_view=allow_one_view,
            min_size=min_size,
            min_extent=min_extent,
            dtype=engine.dtype,
        ):
            print(f"Yielding {class_name} volume of shape {vol.shape}")
            yield vol, class_name, tracker

    # verify that the image doesn't have extraneous channel dimensions
    assert image.ndim in [3, 4], "Only 3D and 4D input images can be handled!"
    if image.ndim == 4:
        # channel dimensions are commonly 1, 3 and 4
        # check for dimensions on zeroeth and last axes
        shape = image.shape
        if shape[0] in [1, 3, 4]:
            image = image[0]
        elif shape[-1] in [1, 3, 4]:
            image = image[..., 0]
        else:
            raise Exception(f"Image volume must be 3D, got image of shape {shape}")

        print(
            f"Got 4D image of shape {shape}, extracted single channel of size {image.shape}"
        )

    if orthoplane:
        trackers_dict = orthoplane_inference(engine, image)
        return start_consensus_worker(trackers_dict)
    else:
        outputs = stack_inference(engine, image, inference_plane)
        return start_postprocess_worker(*outputs)


def stack_postprocessing(
    trackers,
    model_config,
    label_divisor=1000,
    min_size=200,
    min_extent=4,
    dtype=np.uint32,
):
    
    thing_list = model_config["thing_list"]
    class_names = model_config["class_names"]

    # create the final instance segmentations
    for class_id, class_name in class_names.items():
        print(f"Creating stack segmentation for class {class_name}...")

        class_tracker = get_axis_trackers_by_class(trackers, class_id)[0]
        shape3d = class_tracker.shape3d

        # merge instances from orthoplane inference
        stack_tracker = InstanceTracker(class_id, label_divisor, shape3d, "xy")
        stack_tracker.instances = instance_relabel(class_tracker)

        # inplace apply filters to final merged segmentation
        if class_id in thing_list:
            filters.remove_small_objects(stack_tracker, min_size=min_size)
            filters.remove_pancakes(stack_tracker, min_span=min_extent)

        print(f"Total {class_name} objects {len(stack_tracker.instances.keys())}")

        # decode and fill the instances
        stack_vol = np.zeros(shape3d, dtype=dtype)

        fill_volume(stack_vol, stack_tracker.instances)

        yield stack_vol, class_name, stack_tracker.instances


def tracker_consensus(
    trackers,
    model_config,
    pixel_vote_thr=2,
    cluster_iou_thr=0.75,
    allow_one_view=False,
    min_size=200,
    min_extent=4,
    dtype=np.uint32,
):
    
    labels = model_config["labels"]
    thing_list = model_config["thing_list"]
    class_names = model_config["class_names"]

    # create the final instance segmentations
    for class_id, class_name in class_names.items():
        # get the relevant trackers for the class_label
        print(f"Creating consensus segmentation for class {class_name}...")

        class_trackers = get_axis_trackers_by_class(trackers, class_id)
        shape3d = class_trackers[0].shape3d

        # consensus from orthoplane
        if class_id in thing_list:
            consensus_tracker = create_instance_consensus(
                class_trackers, pixel_vote_thr, cluster_iou_thr, allow_one_view
            )
            filters.remove_small_objects(consensus_tracker, min_size=min_size)
            filters.remove_pancakes(consensus_tracker, min_span=min_extent)
        else:
            consensus_tracker = create_semantic_consensus(
                class_trackers, pixel_vote_thr
            )

        print(f"Total {class_name} objects {len(consensus_tracker.instances.keys())}")

        # decode and fill the instances
        consensus_vol = np.zeros(shape3d, dtype=dtype)

        fill_volume(consensus_vol, consensus_tracker.instances)

        yield consensus_vol, class_name, consensus_tracker.instances
