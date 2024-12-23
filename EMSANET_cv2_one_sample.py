# -*- coding: utf-8 -*-
"""
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import time as time
import pyrealsense2 as rs

from nicr_mt_scene_analysis.data import move_batch_to_device
from nicr_mt_scene_analysis.data import mt_collate

from emsanet.args import ArgParserEMSANet
from emsanet.data import get_datahelper
from emsanet.model import EMSANet
from emsanet.preprocessing import get_preprocessor
from emsanet.visualization import visualize_predictions
from emsanet.weights import load_weights


# Bag file path
BAG_FILE = os.path.join("samples","d435i_sample_data", "d435i_walking.bag")

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
                           dtype=torch.uint8, device=torch.device('cuda'))
    return palette

palette = index_to_color()

def _get_args():
    parser = ArgParserEMSANet()

    group = parser.add_argument_group('Inference')
    group.add_argument('--inference-input-height',
                       type=int,
                       default=480,
                       dest='validation_input_height',
                       help="Network input height for predicting on inference data.")
    group.add_argument('--inference-input-width',
                       type=int,
                       default=640,
                       dest='validation_input_width',
                       help="Network input width for predicting on inference data.")
    group.add_argument('--depth-max',
                       type=float,
                       default=None,
                       help="Additional max depth values.")
    group.add_argument('--depth-scale',
                       type=float,
                       default=1.0,
                       help="Additional depth scaling factor to apply.")

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

    if DEPTH_DO_NOT_FORCE_MM:
        predefined_args.append('--sunrgbd-depth-do-not-force-mm')
    if NO_PRETRAINED_BACKBONE:
        predefined_args.append('--no-pretrained-backbone')
    if RAW_DEPTH:
        predefined_args.append('--raw-depth')

    args = parser.parse_args(predefined_args)
    return args

def start_pipeline(bag_file=BAG_FILE):
    while True:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device_from_file(bag_file, repeat_playback=False)
        profile = pipeline.start(config)
        align = rs.align(rs.stream.color)

        try:
            while True:
                try:
                    frames = pipeline.wait_for_frames()
                except RuntimeError:
                    print("Playback ended. Restarting pipeline.")
                    break

                aligned_frames = align.process(frames)
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()

                if not depth_frame or not color_frame:
                    print("Could not acquire depth or color frames.")
                    break

                yield depth_frame, color_frame
        finally:
            pipeline.stop()

def _load_img(fp):
    img = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Image file not found or unable to load: {fp}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def anything_mask(predictions, classes_str):
    classes = classes_str.split()
    classes_tensor = torch.tensor([classes_dict()[c] for c in classes],
                                  device=predictions['semantic_segmentation_idx_fullres'][0].device)
    mask = torch.isin(predictions['semantic_segmentation_idx_fullres'][0], classes_tensor)
    return mask

def predictions_to_color(predictions, palette=palette):
    segmentation_idx = predictions['semantic_segmentation_idx_fullres'][0].long()
    color_img = palette[segmentation_idx]
    return color_img.cpu().numpy()

def floor_mask(predictions):
    floor_classes = torch.tensor([1, 19], device=predictions['semantic_segmentation_idx_fullres'][0].device)
    mask = torch.isin(predictions['semantic_segmentation_idx_fullres'][0], floor_classes)
    return mask

def load_model_and_preprocessor(args, device):
    data = get_datahelper(args)
    dataset_config = data.dataset_config
    model = EMSANet(args, dataset_config=dataset_config)

    print(f"Loading checkpoint: '{args.weights_filepath}'")
    checkpoint = torch.load(args.weights_filepath, map_location=torch.device('cpu'))
    state_dict = checkpoint.get('state_dict', checkpoint)
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

def preprocess_sample(args, preprocessor, device, fp_rgb, fp_depth):
    """
    Preprocess the input RGB and depth images into a batch ready for model inference.
    """
    img_rgb = _load_img(fp_rgb)
    img_depth = _load_img(fp_depth).astype('float32')

    if args.depth_max is not None:
        img_depth[img_depth > args.depth_max] = 0
    img_depth *= args.depth_scale

    sample = preprocessor({
        'rgb': img_rgb,
        'depth': img_depth,
        'identifier': os.path.basename(os.path.splitext(fp_rgb)[0])
    })

    batch = mt_collate([sample])
    batch = move_batch_to_device(batch, device=device)
    return img_rgb, batch

def run_model_inference(model, batch):
    """
    Run the model inference on the preprocessed batch and return predictions.
    """
    predictions = model(batch, do_postprocessing=True)
    return predictions

def visualize_inference_results(img_rgb, predictions):
    """
    Visualize the predictions on the original RGB image.
    """
    semantic_seg = predictions_to_color(predictions)
    img_rgb_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    img_semantic_bgr = cv2.cvtColor(semantic_seg, cv2.COLOR_RGB2BGR)

    cv2.namedWindow('RGB and Semantic Segmentation', cv2.WINDOW_NORMAL)
    img_overlay = cv2.addWeighted(img_rgb_bgr, 0.3, img_semantic_bgr, 0.7, 0)
    cv2.imshow('RGB and Semantic Segmentation', img_overlay)
    cv2.waitKey(0)

def main():
    args = _get_args()
    device = torch.device(args.device)
    model, preprocessor = load_model_and_preprocessor(args, device)

    basepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'samples')
    fp_rgb = os.path.join(basepath, 'sample_rgb.png')
    fp_depth = os.path.join(basepath, 'sample_depth.png')

    try:
        img_rgb, batch = preprocess_sample(args, preprocessor, device, fp_rgb, fp_depth)
        predictions = run_model_inference(model, batch)
        visualize_inference_results(img_rgb, predictions)
    except Exception as e:
        print(f"An error occurred during processing: {e}")

if __name__ == '__main__':
    main()
