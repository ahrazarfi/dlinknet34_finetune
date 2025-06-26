# DeepGlobe Road Extraction Challenge - DLinkNet34 Finetuning

This project provides a comprehensive framework for finetuning a pretrained DLinkNet34 model on the DeepGlobe road extraction dataset. The model performs semantic segmentation to extract road networks from satellite imagery with configurable training strategies and robust inference capabilities.

## Overview

DLinkNet34 is a deep learning architecture that combines ResNet34 encoder with dilated convolution blocks and skip connections for precise road segmentation. This implementation includes:

- **Pretrained Model**: DLinkNet34 trained on DeepGlobe dataset
- **Finetuning Framework**: Multiple training strategies with progressive layer unfreezing
- **Data Pipeline**: Robust data loading with comprehensive augmentations
- **Inference Engine**: Test-time augmentation (TTA) with multi-view predictions
- **Experiment Tracking**: Weights & Biases (WandB) integration
- **Multi-resolution Support**: 6cm, 30cm, and 50cm per pixel imagery

## Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd DeepGlobe-Road-Extraction-Challenge

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

Your dataset should follow this structure:
```
dataset/custom/train/
├── 123_sat.jpg     # Satellite images
├── 123_mask.png    # Road masks (binary)
├── 456_sat.jpg
├── 456_mask.png
└── ...
```

### 3. Basic Training

```bash
# Run with default gradual finetuning preset
python run_finetune.py --data dataset/custom/train

# Use specific finetuning strategy
python run_finetune.py --data dataset/custom/train --preset aggressive

# Enable WandB logging
python run_finetune.py --data dataset/custom/train --wandb --project my-road-project
```

### 4. Inference

```bash
# Edit test.py to set your input/output paths
python test.py
```

## Finetuning Strategies

The framework provides 5 different finetuning presets in `finetune_configs.py`:

| Preset | Description | Learning Rate | Epochs | Unfreeze Schedule |
|--------|-------------|---------------|--------|-------------------|
| **conservative** | Slow, stable training | 1e-4 | 50 | Epochs 10→20→35 |
| **aggressive** | Fast convergence | 2e-4 | 40 | Epochs 3→8→15 |
| **gradual** | Balanced approach | 2e-4 | 60 | Epochs 5→15→25 |
| **decoder_only** | Train decoder only | 2e-4 | 30 | No unfreezing |
| **full** | Full model training | 1e-4 | 30 | Epoch 1 |

### Unfreeze Schedule
- **Level 1**: Decoder layers
- **Level 2**: Encoder layers 3-4
- **Level 3**: Full model

## Configuration Options

### Command Line Arguments

```bash
python run_finetune.py \
    --data /path/to/dataset \           # Required: path to training data
    --preset conservative \             # Finetuning strategy
    --weights weights/log01_dink34.th \ # Pretrained weights
    --epochs 50 \                       # Override default epochs
    --wandb \                          # Enable WandB logging
    --project my-experiment            # WandB project name
```

### Configurable Parameters

Key parameters can be modified in the training script:

- `batch_per_gpu`: Batch size per GPU (default: 4)
- `val_frac`: Validation split fraction (default: 0.2)
- `img_log_freq`: Image logging frequency for WandB (default: 5 epochs)
- `learning_rate`: Base learning rate (preset-dependent)
- `total_epochs`: Training duration (preset-dependent)

### Data Augmentation

The pipeline includes comprehensive augmentations via Albumentations:

- Random rotations (90° steps)
- Horizontal/vertical flips
- Brightness/contrast adjustments
- Random fog and shadows
- Shift/scale/rotate transformations
- ImageNet normalization

## Weights & Biases Integration

### Setup WandB

1. **Install WandB** (included in requirements.txt)
2. **Login and get API key**:
   ```bash
   wandb login
   ```
3. **Set API key as environment variable**:
   ```bash
   export WANDB_API_KEY=your_api_key_here
   ```
   
   Or add to your `.bashrc`/`.zshrc`:
   ```bash
   echo 'export WANDB_API_KEY=your_api_key_here' >> ~/.bashrc
   source ~/.bashrc
   ```

### Using WandB

```bash
# Enable WandB logging with custom project name
python run_finetune.py --data dataset/custom/train --wandb --project road-extraction-v2

# WandB will track:
# - Training/validation losses
# - Dice coefficient and IoU metrics
# - Learning rate changes
# - Model gradients
# - Sample predictions every 5 epochs
# - Model artifacts (best checkpoints)
```

## Model Architecture

**DLinkNet34** combines:
- **Encoder**: ResNet34 backbone (ImageNet pretrained)
- **Dilated Center**: Multi-scale dilated convolutions (1x, 2x, 4x, 8x)
- **Decoder**: Progressive upsampling with skip connections
- **Output**: Single-channel logits for binary road segmentation

## Inference Features

The `test.py` script provides robust inference with:

### Test-Time Augmentation (TTA)
- **1-view**: No augmentation
- **2-view**: 90° rotation
- **4-view**: Rotations + horizontal flip
- **8-view**: Rotations + horizontal/vertical flips

### Multi-format Support
- **Input**: GeoTIFF, PNG, JPG with any bit depth
- **Output**: Binary road masks (PNG format)
- **Automatic radiometric matching** to DeepGlobe training data

### Configurable Parameters
```python
THRESHOLD = 0.3          # Probability threshold for binary masks
TARGET_SIZE = 1024       # Input image size (matches training)
BATCHSIZE_PER_CARD = 4   # Batch size per GPU
```

## File Structure

```
DeepGlobe-Road-Extraction-Challenge/
├── run_finetune.py         # Main training script
├── train_finetune.py       # Training loop implementation  
├── finetune_configs.py     # Finetuning strategy presets
├── test.py                 # Inference script with TTA
├── networks/
│   └── dinknet.py         # DLinkNet34 model architecture
├── roads_dataset.py        # Dataset class with augmentations
├── framework.py            # Training framework utilities
├── losses.py              # Loss functions (Combo loss)
├── requirements.txt        # Python dependencies
├── weights/               # Model checkpoints
├── DATA/                  # Multi-resolution datasets
└── logs/                  # Training logs
```

## Performance Tips

1. **GPU Memory**: Default batch size is 4 per GPU. Reduce if needed.
2. **Data Loading**: Use SSD storage for faster I/O during training.
3. **Validation**: 20% validation split by default. Adjust with `--val_frac`.
4. **Early Stopping**: Training stops automatically when LR < 5e-7.
5. **TTA Speed**: 8-view TTA provides best accuracy but is 8x slower than single view.

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce `batch_per_gpu` in the training script
2. **WandB not logging**: Check API key setup and internet connection
3. **Data loading errors**: Verify dataset structure and file paths
4. **Slow training**: Ensure data is on SSD and use adequate GPU memory

### Model Weights

- Default pretrained weights: `weights/log01_dink34.th`
- Best model saved as: `weights/{experiment_name}_best.th`
- Models automatically uploaded to WandB as artifacts

## License

This project is licensed under the terms specified in the LICENSE file.

## Citation

If you use this code in your research, please cite the original DLinkNet paper and the DeepGlobe dataset.