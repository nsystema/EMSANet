# -*- coding: utf-8 -*-
"""
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import os

import cv2
import matplotlib.pyplot as plt
import torch
import time as time

from nicr_mt_scene_analysis.data import move_batch_to_device
from nicr_mt_scene_analysis.data import mt_collate

from emsanet.args import ArgParserEMSANet
from emsanet.data import get_datahelper
from emsanet.model import EMSANet
from emsanet.preprocessing import get_preprocessor
from emsanet.visualization import visualize_predictions
from emsanet.weights import load_weights

def classes_dict():
    return {
        'wall': 0,
        'floor': 1,
        'cabinet': 2,
        'bed': 3,
        'chair': 4,
        'sofa': 5,
        'table': 6,
        'door': 7,
        'window': 8,
        'bookshelf': 9,
        'picture': 10,
        'counter': 11,
        'blinds': 12,
        'desk': 13,
        'shelves': 14,
        'curtain': 15,
        'dresser': 16,
        'pillow': 17,
        'mirror': 18,
        'floor_mat': 19,
        'clothes': 20,
        'ceiling': 21,
        'books': 22,
        'fridge': 23,
        'tv': 24,
        'paper': 25,
        'towel': 26,
        'shower_curtain': 27,
        'box': 28,
        'whiteboard': 29,
        'person': 30,
        'night_stand': 31,
        'toilet': 32,
        'sink': 33,
        'lamp': 34,
        'bathtub': 35,
        'bag': 36
    }


def _get_args():
    """
    Returns a Namespace object with predefined arguments for semantic segmentation inference.
    This replaces the need for command-line argument parsing.
    """
    parser = ArgParserEMSANet()

    # Add additional arguments specific to inference
    group = parser.add_argument_group('Inference')
    group.add_argument(    # useful for appm context module
        '--inference-input-height',
        type=int,
        default=480,
        dest='validation_input_height',    # used in test phase
        help="Network input height for predicting on inference data."
    )
    group.add_argument(    # useful for appm context module
        '--inference-input-width',
        type=int,
        default=640,
        dest='validation_input_width',    # used in test phase
        help="Network input width for predicting on inference data."
    )
    group.add_argument(
        '--depth-max',
        type=float,
        default=None,
        help="Additional max depth values. Values above are set to zero as "
             "they are most likely not valid. Note, this clipping is applied "
             "before scaling the depth values."
    )
    group.add_argument(
        '--depth-scale',
        type=float,
        default=1.0,
        help="Additional depth scaling factor to apply."
    )

    # Define the predefined arguments as a list, focusing only on semantic segmentation
    predefined_args = [
        '--dataset', 'sunrgbd',
        '--sunrgbd-depth-do-not-force-mm',
        '--tasks', 'semantic',  # Only semantic task
        # '--enable-panoptic',  # Removed since panoptic includes other tasks
        '--rgb-encoder-backbone', 'resnet34',
        '--rgb-encoder-backbone-block', 'nonbottleneck1d',
        '--depth-encoder-backbone', 'resnet34',
        '--depth-encoder-backbone-block', 'nonbottleneck1d',
        '--no-pretrained-backbone',
        '--input-modalities', 'rgb', 'depth',
        '--raw-depth',
        '--depth-max', '8000',
        '--depth-scale', '8',
        '--weights-filepath', './trained_models/sunrgbd/r34_NBt1D.pth'
    ]

    # Parse the predefined arguments
    args = parser.parse_args(predefined_args)

    return args


def _load_img(fp):


    """
    Loads an image from the specified file path.

    Args:
        fp (str): Path to the image file.

    Returns:
        numpy.ndarray: Loaded image in RGB format if it's a color image, otherwise unchanged.
    
    Raises:
        FileNotFoundError: If the image file does not exist or cannot be loaded.
    """
    img = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Image file not found or unable to load: {fp}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def anything_mask(predictions, classes_str):
    """
    Create a mask for the specified classes by checking if the predicted semantic segmentation indices
    Args:
        predictions: The predictions dictionary from the model
        classes_str: The classes to include in the mask
    Returns:
        mask: A mask of the specified classes
    """
    classes = classes_str.split()
    classes_tensor = torch.tensor([classes_dict()[c] for c in classes], device=predictions['semantic_segmentation_idx_fullres'][0].device)
    mask = torch.isin(predictions['semantic_segmentation_idx_fullres'][0], classes_tensor)
    return mask

# example usage
# mask = anything_mask(predictions, "floor wall")

def floor_mask(predictions):
    # Create a mask for the floor by checking if the predicted semantic segmentation indices
    floor_classes = torch.tensor([1, 19], device=predictions['semantic_segmentation_idx_fullres'][0].device)
    mask = torch.isin(predictions['semantic_segmentation_idx_fullres'][0], floor_classes)
    return mask

def main():
    args = _get_args()
    assert all(x in args.input_modalities for x in ('rgb', 'depth')), \
        "Only RGBD inference supported so far"

    device = torch.device(args.device)

    # Data and model
    data = get_datahelper(args)
    dataset_config = data.dataset_config
    model = EMSANet(args, dataset_config=dataset_config)

    # Load weights
    print(f"Loading checkpoint: '{args.weights_filepath}'")
    checkpoint = torch.load(args.weights_filepath,
                            map_location=torch.device('cpu'))
    state_dict = checkpoint.get('state_dict', checkpoint)  # Handle cases without 'state_dict' key
    if 'epoch' in checkpoint:
        print(f"-> Epoch: {checkpoint['epoch']}")
    load_weights(args, model, state_dict, verbose=False)

    torch.set_grad_enabled(False)
    model.eval()
    model.to(device)

    # Build preprocessor
    preprocessor = get_preprocessor(
        args,
        dataset=data.datasets_valid[0],
        phase='test',
        multiscale_downscales=None
    )

    # Define specific sample paths
    basepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'samples')
    fp_rgb = os.path.join(basepath, 'sample_rgb.png')
    fp_depth = os.path.join(basepath, 'sample_depth.png')

    # Verify that both files exist
    if not os.path.isfile(fp_rgb):
        raise FileNotFoundError(f"RGB image file not found: {fp_rgb}")
    if not os.path.isfile(fp_depth):
        raise FileNotFoundError(f"Depth image file not found: {fp_depth}")

    # Process the single pair of images
    try:
        t0 = time.time()
        # Load RGB and Depth images
        img_rgb = _load_img(fp_rgb)
        img_depth = _load_img(fp_depth).astype('float32')

        if args.depth_max is not None:
            img_depth[img_depth > args.depth_max] = 0
        img_depth *= args.depth_scale

        # Preprocess sample
        sample = preprocessor({
            'rgb': img_rgb,
            'depth': img_depth,
            'identifier': os.path.basename(os.path.splitext(fp_rgb)[0])
        })

        # Add batch axis as there is no dataloader
        batch = mt_collate([sample])
        batch = move_batch_to_device(batch, device=device)

        preprocess_time = time.time() - t0

        # Apply model
        predictions = model(batch, do_postprocessing=True)
        inference_time = time.time() - t0 - preprocess_time
        # Visualize predictions
        preds_viz = visualize_predictions(
            predictions=predictions,
            batch=batch,
            dataset_config=dataset_config
        )
        semantic_seg = preds_viz.get('semantic_segmentation_idx_fullres', [])[0]
        all_vis_time = time.time() - inference_time 

        # generate the floor mask
        mask = floor_mask(predictions)
        floor_mask_time = time.time() - all_vis_time
    

        # Show results - Only Semantic Segmentation
        plt.figure(figsize=(12, 6), dpi=150)

        # Display RGB Image
        plt.subplot(1, 3, 1)
        plt.title('RGB')
        plt.imshow(img_rgb)
        plt.axis('off')

        # Display Semantic Segmentation
        plt.subplot(1, 3, 2)
        plt.title('Semantic Segmentation')
        if semantic_seg is not None:
            plt.imshow(semantic_seg, interpolation='nearest')
        else:
            print("Warning: 'semantic_segmentation_idx_fullres' not found in predictions.")
            plt.text(0.5, 0.5, 'No Semantic Segmentation Output', horizontalalignment='center',
                     verticalalignment='center', transform=plt.gca().transAxes)
        plt.axis('off')

        # Display Floor Mask
        plt.subplot(1, 3, 3)
        plt.title('Floor Mask')
        plt.imshow(mask.cpu().numpy(), cmap='gray')
        plt.axis('off')

        # Set the main title
        scene = preds_viz.get('scene', ['Unknown'])[0]
        model_name = os.path.basename(args.weights_filepath)
        image_name = os.path.basename(fp_rgb)
        plt.suptitle(
            f"Image: {image_name}, "
            f"Model: {model_name}, "
            f"Scene: {scene}"
        )

        plt.figtext(0.5, 0.01, f"Preprocessing: {preprocess_time * 1000:.2g} ms, "
                       f"Inference: {inference_time * 1000:.2g} ms, "
                       f"Visualization: {all_vis_time * 1000:.2g} ms, "
                       f"Floor Mask: {floor_mask_time * 1000:.2g} ms",
                ha='center', va='center')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])


        plt.show()

    except Exception as e:
        print(f"An error occurred during processing: {e}")


if __name__ == '__main__':
    main()
