# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the 1st place solution for the DeepGlobe Road Extraction Challenge, implementing semantic segmentation for satellite road extraction using PyTorch. The project uses D-LinkNet34 as the primary architecture with Test Time Augmentation (TTA) for improved performance.

## Common Commands

### Training
```bash
python train.py
```
Trains the default D-LinkNet34 model. Training parameters:
- Default batch size: 4 per GPU
- Learning rate: 2e-4 with adaptive reduction
- Early stopping after 6 epochs without improvement
- Model saves to `weights/log01_dink34.th`

### Inference
```bash
python test.py
```
Runs inference with TTA on test images. The script automatically selects TTA strategy based on available GPU memory.

### Custom Dataset Testing
```bash
python test_custom.py
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