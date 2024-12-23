# -*- coding: utf-8 -*-
"""
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import os
import cv2
import numpy as np
import torch
import pyrealsense2 as rs

from nicr_mt_scene_analysis.data import move_batch_to_device, mt_collate

from emsanet.args import ArgParserEMSANet
from emsanet.data import get_datahelper
from emsanet.model import EMSANet
from emsanet.preprocessing import get_preprocessor
from emsanet.visualization import visualize_predictions
from emsanet.weights import load_weights



# Bag file path
BAG_FILE = "samples\d435i_sample_data\my_room_3.bag"

# Dataset Configuration
DATASET = 'sunrgbd'
DEPTH_DO_NOT_FORCE_MM = True  # Boolean flag, presence indicated by the flag

# Task Configuration
TASKS = 'semantic'

# Encoder Configurations
RGB_ENCODER_BACKBONE = 'resnet34'
RGB_ENCODER_BACKBONE_BLOCK = 'nonbottleneck1d'
DEPTH_ENCODER_BACKBONE = 'resnet34'
DEPTH_ENCODER_BACKBONE_BLOCK = 'nonbottleneck1d'

# Additional Configurations
NO_PRETRAINED_BACKBONE = True  # Boolean flag
INPUT_MODALITIES = ['rgb', 'depth']
RAW_DEPTH = True  # Boolean flag
DEPTH_MAX = '8000'
DEPTH_SCALE = '8'
WEIGHTS_FILEPATH = './trained_models/sunrgbd/r34_NBt1D.pth'

# Color Mapping
FLOOR_CEILING_ONLY = True  # If True color mapping will consider only floor and floor mat as floor class and merge the rest into a single class
CUSTOM_SEGMENTATION = False # If True color mapping will consider floor (merged with floor mat), wall, and ceiling as seperate classes and merge the rest into a single class





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


def index_to_color():
    color_dict = {
        0: (255, 0, 0),               # wall - Red
        1: (255, 165, 0),             # floor - Orange
        2: (255, 255, 0),             # cabinet - Yellow
        3: (0, 128, 0),               # bed - Green
        4: (0, 255, 255),             # chair - Cyan
        5: (0, 0, 255),               # sofa - Blue
        6: (128, 0, 128),             # table - Purple
        7: (255, 192, 203),           # door - Pink
        8: (128, 128, 128),           # window - Gray
        9: (165, 42, 42),             # bookshelf - Brown
        10: (255, 20, 147),           # picture - Deep Pink
        11: (0, 255, 0),              # counter - Lime
        12: (34, 139, 34),            # blinds - Forest Green
        13: (70, 130, 180),           # desk - Steel Blue
        14: (123, 104, 238),          # shelves - Medium Slate Blue
        15: (255, 105, 180),          # curtain - Hot Pink
        16: (154, 205, 50),           # dresser - Yellow Green
        17: (255, 182, 193),          # pillow - Light Pink
        18: (192, 192, 192),          # mirror - Silver
        19: (160, 82, 45),            # floor_mat - Sienna
        20: (255, 69, 0),             # clothes - Orange Red
        21: (238, 130, 238),          # ceiling - Violet
        22: (0, 100, 0),              # books - Dark Green
        23: (0, 0, 128),              # fridge - Navy
        24: (255, 0, 255),            # tv - Magenta
        25: (255, 255, 224),          # paper - Light Yellow
        26: (0, 255, 127),            # towel - Spring Green
        27: (0, 206, 209),            # shower_curtain - Dark Turquoise
        28: (255, 140, 0),            # box - Dark Orange
        29: (0, 0, 0),                # whiteboard - Black
        30: (255, 215, 0),            # person - Gold
        31: (34, 139, 34),            # night_stand - Forest Green
        32: (220, 20, 60),            # toilet - Crimson
        33: (75, 0, 130),             # sink - Indigo
        34: (64, 224, 208),           # lamp - Turquoise
        35: (176, 196, 222),          # bathtub - Light Steel Blue
        36: (199, 21, 133)            # bag - Medium Violet Red
    }
    num_classes = max(color_dict.keys()) + 1

    palette = torch.tensor([color_dict.get(i, (0, 0, 0)) for i in range(num_classes)],
                           dtype=torch.uint8, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    return palette


def index_to_color_custom():
    color_dict = {
        0: (255, 0, 0),               # wall - Red
        1: (167, 221, 154),           # floor - Green
        2: (0, 0, 0),                 # cabinet - Black
        3: (0, 0, 0),                 # bed - Black
        4: (0, 0, 0),                 # chair - Black
        5: (0, 0, 0),                 # sofa - Black
        6: (0, 0, 0),                 # table - Black
        7: (0, 0, 0),                 # door - Black
        8: (0, 0, 0),                 # window - Black
        9: (0, 0, 0),                 # bookshelf - Black
        10:(0, 0, 0),                 # picture - Black
        11:(0, 0, 0),                 # counter - Black
        12:(0, 0, 0),                 # blinds - Black
        13:(0, 0, 0),                 # desk - Black
        14:(0, 0, 0),                 # shelves - Black
        15:(0, 0, 0),                 # curtain - Black
        16:(0, 0, 0),                 # dresser - Black
        17:(0, 0, 0),                 # pillow - Black
        18:(0, 0, 0),                 # mirror - Black
        19:(167, 221, 154),           # floor_mat - Green
        20:(0, 0, 0),                 # clothes - Black
        21:(100,151,177),             # ceiling - Blue
        22:(0, 0, 0),                 # books - Black
        23:(0, 0, 0),                 # fridge - Black
        24:(0, 0, 0),                 # tv - Black
        25:(0, 0, 0),                 # paper - Black
        26:(0, 0, 0),                 # towel - Black
        27:(0, 0, 0),                 # shower_curtain - Black
        28:(0, 0, 0),                 # box - Black
        29:(0, 0, 0),                 # whiteboard - Black
        30:(0, 0, 0),                 # person - Black
        31:(0, 0, 0),                 # night_stand - Black
        32:(0, 0, 0),                 # toilet - Black
        33:(0, 0, 0),                 # sink - Black
        34:(0, 0, 0),                 # lamp - Black
        35:(0, 0, 0),                 # bathtub - Black
        36:(0, 0, 0)                  # bag - Black
    }
    num_classes = max(color_dict.keys()) + 1

    palette = torch.tensor([color_dict.get(i, (0, 0, 0)) for i in range(num_classes)],
                            dtype=torch.uint8, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    return palette

def index_to_color_floor_ceiling():
    color_dict = {
        0: (0, 0, 0),                 # wall - Black
        1: (167, 221, 154),           # floor - Green
        2: (0, 0, 0),                 # cabinet - Black
        3: (0, 0, 0),                 # bed - Black
        4: (0, 0, 0),                 # chair - Black
        5: (0, 0, 0),                 # sofa - Black
        6: (0, 0, 0),                 # table - Black
        7: (0, 0, 0),                 # door - Black
        8: (0, 0, 0),                 # window - Black
        9: (0, 0, 0),                 # bookshelf - Black
        10:(0, 0, 0),                 # picture - Black
        11:(0, 0, 0),                 # counter - Black
        12:(0, 0, 0),                 # blinds - Black
        13:(0, 0, 0),                 # desk - Black
        14:(0, 0, 0),                 # shelves - Black
        15:(0, 0, 0),                 # curtain - Black
        16:(0, 0, 0),                 # dresser - Black
        17:(0, 0, 0),                 # pillow - Black
        18:(0, 0, 0),                 # mirror - Black
        19:(167, 221, 154),           # floor_mat - Green
        20:(0, 0, 0),                 # clothes - Black
        21:(167, 221, 154),           # ceiling - Green
        22:(0, 0, 0),                 # books - Black
        23:(0, 0, 0),                 # fridge - Black
        24:(0, 0, 0),                 # tv - Black
        25:(0, 0, 0),                 # paper - Black
        26:(0, 0, 0),                 # towel - Black
        27:(0, 0, 0),                 # shower_curtain - Black
        28:(0, 0, 0),                 # box - Black
        29:(0, 0, 0),                 # whiteboard - Black
        30:(0, 0, 0),                 # person - Black
        31:(0, 0, 0),                 # night_stand - Black
        32:(0, 0, 0),                 # toilet - Black
        33:(0, 0, 0),                 # sink - Black
        34:(0, 0, 0),                 # lamp - Black
        35:(0, 0, 0),                 # bathtub - Black
        36:(0, 0, 0)                  # bag - Black
    }

    num_classes = max(color_dict.keys()) + 1

    palette = torch.tensor([color_dict.get(i, (0, 0, 0)) for i in range(num_classes)],
                            dtype=torch.uint8, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    return palette

if FLOOR_CEILING_ONLY:
    palette = index_to_color_floor_ceiling()
else:
    if CUSTOM_SEGMENTATION:
        palette = index_to_color_custom()
    else:
        palette = index_to_color()


def _get_args():
    """
    Returns a Namespace object with predefined arguments for semantic segmentation inference.
    This replaces the need for command-line argument parsing.
    """
    parser = ArgParserEMSANet()

    # Add additional arguments specific to inference
    group = parser.add_argument_group('Inference')
    group.add_argument(
        '--inference-input-height',
        type=int,
        default=480,
        dest='validation_input_height',    # used in test phase
        help="Network input height for predicting on inference data."
    )
    group.add_argument(
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

    # Construct the predefined arguments list using constants
    predefined_args = [
        '--dataset', DATASET,
        '--tasks', TASKS,
        '--rgb-encoder-backbone', RGB_ENCODER_BACKBONE,
        '--rgb-encoder-backbone-block', RGB_ENCODER_BACKBONE_BLOCK,
        '--depth-encoder-backbone', DEPTH_ENCODER_BACKBONE,
        '--depth-encoder-backbone-block', DEPTH_ENCODER_BACKBONE_BLOCK,
        '--input-modalities', *INPUT_MODALITIES,
        '--depth-max', DEPTH_MAX,
        '--depth-scale', DEPTH_SCALE,
        '--weights-filepath', WEIGHTS_FILEPATH
    ]

    # Add flags if their corresponding constants are True
    if DEPTH_DO_NOT_FORCE_MM:
        predefined_args.append('--sunrgbd-depth-do-not-force-mm')
    if NO_PRETRAINED_BACKBONE:
        predefined_args.append('--no-pretrained-backbone')
    if RAW_DEPTH:
        predefined_args.append('--raw-depth')

    # Parse the predefined arguments
    args = parser.parse_args(predefined_args)

    return args


def start_pipeline(bag_file=BAG_FILE):
    """
    Initializes and starts the RealSense pipeline for reading from a bag file.

    Args:
        bag_file (str): Path to the RealSense bag file.

    Yields:
        tuple: A tuple containing the depth frame and color frame.
    """
    while True:
        # Initialize the RealSense pipeline
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device_from_file(bag_file, repeat_playback=True)
        profile = pipeline.start(config)
        align = rs.align(rs.stream.color)

        try:
            while True:
                try:
                    # Wait for the next set of frames
                    frames = pipeline.wait_for_frames()
                except RuntimeError:
                    # Handle end of playback
                    print("Playback ended. Restarting pipeline.")
                    break

                # Align the frames to the color stream
                aligned_frames = align.process(frames)
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()

                if not depth_frame or not color_frame:
                    print("Could not acquire depth or color frames.")
                    break

                yield depth_frame, color_frame
        finally:
            # Stop the pipeline to release resources
            pipeline.stop()


def _load_img_from_frame(frame, color=True):
    """
    Converts a RealSense frame to a NumPy array.

    Args:
        frame (rs.frame): The RealSense frame.
        color (bool): Whether the frame is a color frame.

    Returns:
        np.ndarray: The converted image.
    """
    if color:
        img = np.asanyarray(frame.get_data())
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = np.asanyarray(frame.get_data())
    return img


def anything_mask(predictions, classes_str):
    """
    Create a mask for the specified classes by checking if the predicted semantic segmentation indices.

    Args:
        predictions (dict): The predictions dictionary from the model.
        classes_str (str): The classes to include in the mask.

    Returns:
        torch.Tensor: A mask of the specified classes.
    """
    classes = classes_str.split()
    classes_tensor = torch.tensor([classes_dict()[c] for c in classes],
                                  device=predictions['semantic_segmentation_idx_fullres'][0].device)
    mask = torch.isin(predictions['semantic_segmentation_idx_fullres'][0], classes_tensor)
    return mask


def predictions_to_color(predictions, palette=palette):
    """
    Convert semantic segmentation indices to a color image efficiently.

    Args:
        predictions (dict): The predictions dictionary from the model. 
                            It should contain the key 'semantic_segmentation_idx_fullres' 
                            with a tensor of shape (1, H, W).
        palette (torch.Tensor): A tensor mapping class indices to RGB colors.

    Returns:
        np.ndarray: The color image of the semantic segmentation with shape (H, W, 3).
    """
    segmentation_idx = predictions['semantic_segmentation_idx_fullres'][0].long()  # Shape: (H, W)
    color_img = palette[segmentation_idx]

    return color_img.cpu().numpy()


def floor_mask(predictions):
    """
    Create a mask for the floor by checking if the predicted semantic segmentation indices.

    Args:
        predictions (dict): The predictions dictionary from the model.

    Returns:
        torch.Tensor: A mask for floor classes.
    """
    floor_classes = torch.tensor([1, 19], device=predictions['semantic_segmentation_idx_fullres'][0].device)
    mask = torch.isin(predictions['semantic_segmentation_idx_fullres'][0], floor_classes)
    return mask


def load_model_and_preprocessor(args, device):
    """
    Load the model and preprocessor using given arguments and device.

    Args:
        args (Namespace): Parsed arguments.
        device (torch.device): The device to load the model on.

    Returns:
        tuple: The loaded model and preprocessor.
    """
    data = get_datahelper(args)
    dataset_config = data.dataset_config
    model = EMSANet(args, dataset_config=dataset_config)

    print(f"Loading checkpoint: '{args.weights_filepath}'")
    checkpoint = torch.load(args.weights_filepath, map_location=torch.device('cpu'))
    state_dict = checkpoint.get('state_dict', checkpoint)  # Handle cases without 'state_dict' key
    if 'epoch' in checkpoint:
        print(f"-> Epoch: {checkpoint['epoch']}")
    load_weights(args, model, state_dict, verbose=False)

    torch.set_grad_enabled(False)
    model.eval()
    model.to(device)

    preprocessor = get_preprocessor(
        args,
        dataset=data.datasets_valid[0],
        phase='test',
        multiscale_downscales=None
    )
    return model, preprocessor


def preprocess_frame(args, preprocessor, device, color_frame, depth_frame):
    """
    Preprocess the input color and depth frames into a batch ready for model inference.

    Args:
        args (Namespace): Parsed arguments.
        preprocessor (Preprocessor): The preprocessor object.
        device (torch.device): The device to move the batch to.
        color_frame (rs.frame): The color frame from the pipeline.
        depth_frame (rs.frame): The depth frame from the pipeline.

    Returns:
        tuple: The original RGB image and the preprocessed batch.
    """
    # Convert RealSense frames to NumPy arrays
    img_rgb = _load_img_from_frame(color_frame, color=True)
    img_depth = _load_img_from_frame(depth_frame, color=False).astype('float32')

    # Apply depth clipping and scaling
    if args.depth_max is not None:
        img_depth[img_depth > args.depth_max] = 0
    img_depth *= args.depth_scale

    # Preprocess the sample
    sample = preprocessor({
        'rgb': img_rgb,
        'depth': img_depth,
        'identifier': 'pipeline_frame'
    })

    # Collate into a batch and move to device
    batch = mt_collate([sample])
    batch = move_batch_to_device(batch, device=device)
    return img_rgb, batch


def run_model_inference(model, batch):
    """
    Run the model inference on the preprocessed batch and return predictions.

    Args:
        model (EMSANet): The loaded EMSANet model.
        batch (dict): The preprocessed batch.

    Returns:
        dict: The model predictions.
    """
    predictions = model(batch, do_postprocessing=True)
    return predictions


def visualize_inference_results(img_rgb, predictions, window_name='RGB and Semantic Segmentation'):


    """
    Visualize the predictions on the original RGB image.

    Args:
        img_rgb (np.ndarray): The original RGB image.
        predictions (dict): The predictions dictionary from the model.
        window_name (str): The name of the display window.
    """
    semantic_seg = predictions_to_color(predictions)
    img_rgb_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    img_semantic_bgr = cv2.cvtColor(semantic_seg, cv2.COLOR_RGB2BGR)

    # Overlay the semantic segmentation on the original image
    img_overlay = cv2.addWeighted(img_rgb_bgr, 0.4, img_semantic_bgr, 0.6, 0)

    # Display the overlayed image
    cv2.imshow(window_name, img_overlay)



def main():
    args = _get_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model, preprocessor = load_model_and_preprocessor(args, device)

    # Initialize the RealSense pipeline
    pipeline = start_pipeline(BAG_FILE)

    # Create a named window for display
    cv2.namedWindow('RGB and Semantic Segmentation', cv2.WINDOW_NORMAL)

    try:
        for depth_frame, color_frame in pipeline:
            # Preprocess the frames
            img_rgb, batch = preprocess_frame(args, preprocessor, device, color_frame, depth_frame)

            # Run inference
            predictions = run_model_inference(model, batch)

            # Visualize the results
            visualize_inference_results(img_rgb, predictions)

            # Check for user input to exit
            if cv2.waitKey(1) & 0xFF == 27:
                print("Exiting inference loop.")
                break

    except KeyboardInterrupt:
        print("Interrupted by user. Exiting...")
    except Exception as e:
        print(f"An error occurred during processing: {e}")
    finally:
        # Destroy all OpenCV windows
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
