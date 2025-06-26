# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the 1st place solution for the DeepGlobe Road Extraction Challenge, implementing semantic segmentation for satellite road extraction using PyTorch. The project uses D-LinkNet34 as the primary architecture with Test Time Augmentation (TTA) for improved performance.

We need to use the pretrained weight from the DlinkNet34 model and finetune it.

## Problem Statement

Our goal is to finetune the pretrained model on rural Indian imagery to extract roads. Presently, the pretrained model is performing reasonably well,
but we need to improve the performance, it is not able to detect complex topography of indian roads which. We have labeled data which of such roads in the same format this model is trained on, that is satellite image and mask. The major difference I can see from the images in the DeepGlobe dataset is that our images contain roads that are not that well defined, end abruptly or get merged abruptly in different roads. This is where the model is struggling the most as of now. It is also struggling to detect individual roads which are somehow not connected to the main road graph in a particular patch.

## Methodology

Make sure that all changes are reviewed before making them. We will be finetuning on a small subset from our custom dataset, you need to devise a
strategy on how to conduct various experiments for our finetuning workflow. We also want to use Weights and Biases for tracking and logging our experiments. This includes tracking various model artifacts like outputs, loss, IOU etc. Also remember to use the same loss and IOU techniques used in
original model.
Upload qualitative panels every few epochs – 15-20 hard-case images with prediction-versus-ground-truth masks that reviewers can toggle in the UI.
Version best checkpoints and full-resolution predictions as W&B artifacts, enabling later diffing and deterministic retrieval.

## about the DLinkNet34 model
In the DeepGlobe Road Extraction Challenge, the raw size of the images and masks provided is 1024×1024, and the roads in most images span the entire image. Still, roads have some natural properties, such as connectivity, complexity, etc. With these attributes in mind, D-Linknet is designed to receive 1024×1024 images as input and retain detailed spatial information. D-linknet can be divided into A, B, C three parts, called encoder, central part and decoder respectively.

D-linknet uses ResNet34, pre-trained on the ImageNet dataset, as its encoder. ResNet34 was originally designed for the classification of 256×256 medium resolution images, but in this challenge the task was to segment roads from 1024×1024 high resolution satellite images. Considering narrowness, connectivity, complexity, and long road spans, it is important to increase the perceived range of features of the central part of the network and retain details. Pooling layer can multiply the felt range of features, but may reduce the resolution of the central feature map and reduce the spatial information. The empty convolution layer may be an ideal alternative to the pooling layer. D-linknet uses several empty convolution layers with skip-connection in the middle.

## Common Commands

### Activating the environment
```bash
source dlink/bin/activate
```
Activates the python environment

### Training
```bash
python3 train.py
```
Trains the default D-LinkNet34 model. Training parameters:
- Default batch size: 4 per GPU
- Learning rate: 2e-4 with adaptive reduction
- Early stopping after 6 epochs without improvement
- Model saves to `weights/log01_dink34.th`

### Inference
```bash
python3 test.py
```
Runs inference with TTA on test images. The script automatically selects TTA strategy based on available GPU memory.

### Custom Dataset Testing
```bash
python3 test_custom.py
```
For testing on custom datasets placed in `dataset/custom/`

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Code Architecture

### Core Components

**MyFrame (framework.py)**: Main training/inference wrapper
- Handles model initialization with DataParallel
- Manages optimizer (Adam), loss computation, and learning rate scheduling
- Provides methods for training (`optimize()`), saving/loading models, and single image inference

**TTAFrame (test.py)**: Test Time Augmentation inference class
- Implements multiple TTA strategies based on GPU memory availability
- Uses 8 augmentations: rotation (90°), horizontal/vertical flips, and combinations
- Automatically selects batch processing strategy (`test_one_img_from_path_1/2/4/8`)

**ImageFolder (data.py)**: Custom dataset loader
- Applies data augmentation: HSV shifts, geometric transformations, flips, rotations
- Expects paired satellite images (`*_sat.jpg`) and masks (`*_mask.png`)
- Normalizes images to [-1.6, 1.6] range and masks to binary [0,1]

**Loss Function (loss.py)**: Combined Dice + BCE loss
- `dice_bce_loss`: Combines Binary Cross Entropy with Soft Dice Loss
- Optimized for binary segmentation tasks

### Network Architectures (networks/)

- **dinknet.py**: D-LinkNet variants (DinkNet34, DinkNet50, DinkNet101)
- **unet.py**: Standard U-Net implementation  
- **dunet.py**: Dilated U-Net variant
- Primary model: DinkNet34 (LinkNet with dilated convolutions)

### Data Structure

Expected directory structure:
```
dataset/
  train/
    {id}_sat.jpg     # Satellite images
    {id}_mask.png    # Ground truth masks
  valid/
  test/
  custom/            # Custom test images
```

### Key Configuration

- **Image size**: 1024x1024 pixels
- **Normalization**: Images scaled to [-1.6, 1.6], masks to [0,1]
- **Threshold**: Test predictions use 0.2 threshold (test.py:172)
- **Model naming**: Uses 'log01_dink34' as default experiment name

### Training Workflow

1. Load paired satellite/mask images from `dataset/train/`
2. Apply random augmentations (HSV, geometric transforms, flips)
3. Train with combined Dice+BCE loss
4. Monitor training loss with early stopping
5. Reduce learning rate when loss plateaus
6. Save best model to `weights/`

### Inference Workflow

1. Load trained model weights
2. Apply TTA with 8 augmentations per image
3. Average predictions across augmentations
4. Apply threshold and save binary masks
5. Output to `submits/` directory

## GPU Memory Management

The TTA system automatically adapts to available GPU memory:
- 8+ GPUs: Full 8-augmentation TTA
- 4+ GPUs: 4-augmentation strategy  
- 2+ GPUs: 2-augmentation strategy
- 1 GPU: Single batch processing

## File Naming Conventions

- Training logs: `logs/{experiment_name}.log`
- Model weights: `weights/{experiment_name}.th`
- Predictions: `submits/{experiment_name}/`
- Input images: `{id}_sat.jpg` (satellite), `{id}_mask.png` (ground truth)