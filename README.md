# Aerial Images Segmentation

A deep learning project for semantic segmentation of aerial imagery using U-Net architecture. This system can classify aerial images into 6 categories: Building, Land, Road, Vegetation, Water, and Unlabeled areas.

## Table of Contents
- [Dataset](#dataset)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Inference](#inference)
- [Configuration](#configuration)
- [Results](#results)

## Dataset

This project uses the **Semantic Segmentation of Aerial Imagery** dataset from Kaggle, featuring high-resolution satellite imagery of Dubai, UAE.

### Dataset Details
- **Source**: Mohammed Bin Rashid Space Center (MBRSC) satellites
- **Location**: Dubai, UAE
- **Total Images**: 72 images organized into 8 tiles
- **Annotations**: Pixel-wise semantic segmentation labels
- **Classes**: 6 semantic classes with specific color coding

### Class Definitions
| Class | Color Code | Description |
|-------|------------|-------------|
| Building | #3C1098 | Urban structures and buildings |
| Land | #8429F6 | Unpaved areas and bare land |
| Road | #6EC1E4 | Roads and paved surfaces |
| Vegetation | #FEDD3A | Trees, grass, and plant life |
| Water | #E2A929 | Water bodies and aquatic areas |
| Unlabeled | #9B9B9B | Unclassified regions |

### Data Splits
- **Training**: 49 samples (70%)
- **Validation**: 15 samples (20%)
- **Test**: 8 samples (10%)

### Download
Dataset available on Kaggle: [[Semantic Segmentation of Aerial Imagery](https://www.kaggle.com/datasets/humansintheloop/semantic-segmentation-of-aerial-imagery/data)]


## Features

- **Complete training pipeline** with proper train/validation/test splits
- **Production-ready inference** with overlapping patch strategy
- **Comprehensive evaluation** with detailed metrics and visualizations
- **Flexible configuration system** with centralized settings
- **Multiple output formats** including predictions, overlays, and confidence maps
- **Batch processing** for large-scale inference
- **Professional logging** with TensorBoard integration
- **Modular architecture** following software engineering practices

## Project Structure

```
Aerial-Images-Segmentation/
├── main.py                     # Training and evaluation entry point
├── predict.py                  # Inference script
├── requirements.txt            # Project dependencies
├── README.md                   # This file
│
├── src/                        # Source code
│   ├── config/
│   │   └── settings.py         # Configuration management
│   ├── data/
│   │   ├── loaders.py          # Data loading utilities
│   │   └── preprocessors.py    # Image preprocessing
│   ├── models/
│   │   ├── architectures/      # Model definitions
│   │   ├── base.py             # Base model class
│   │   └── factory.py          # Model factory
│   ├── training/
│   │   ├── components.py       # Training components (callbacks, metrics)
│   │   ├── pipeline.py         # Training data pipeline
│   │   └── trainer.py          # Main training orchestration
│   ├── evaluation/
│   │   └── evaluator.py        # Model evaluation
│   └── inference/
│       ├── __init__.py
│       └── predictor.py        # Production inference
│
└── data/                       # Data directory (not in repo)
    ├── Tile 1/
    │   ├── images/
    │   └── masks/
    └── Tile 2/
        ├── images/
        └── masks/
```

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended)

### Setup

1. **Clone the repository**
   ```bash
   git clone git@github.com:rab306/Aerial-Images-Segmentation.git
   cd Aerial-Images-Segmentation
   ```

2. **Create virtual environment**
   ```bash
   python -m venv seg-venv
   source seg-venv/bin/activate  # Linux/Mac
   # seg-venv\Scripts\activate    # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare data structure**
   ```
   data/
   ├── Tile 1/
   │   ├── images/
   │   └── masks/
   └── Tile 2/
       ├── images/
       └── masks/
   ```

## Quick Start

### Training a Model
```bash
python main.py --data_dir ./data --epochs 100 --batch_size 16 --output_dir results

```

### Running Inference
```bash
python predict.py \
    --model_path path/to/model.h5 \
    --input_path path/to/aerial/image.jpg \
    --output_path predictions/ \
    --save_visualizations
```

### Evaluating a Trained Model
```bash
python main.py \
    --data_dir ./data \
    --evaluate_only \
    --model_path path/to/model.h5 \
    --dataset test
```

## Usage

### Training

The training script supports various options:

```bash
python main.py [OPTIONS]

Required:
  --data_dir PATH              Path to data directory containing Tile folders

Optional:
  --output_dir PATH            Output directory (default: output)
  --epochs INT                 Number of training epochs (default: 500)
  --batch_size INT             Batch size (default: 32)
  --patch_size INT             Patch size (default: 256)
  --model_type {unet}          Model architecture (default: unet)
  --no_save                    Don't save the trained model
```

### Inference

The inference script supports both single images and batch processing:

```bash
python predict.py [OPTIONS]

Required:
  --model_path PATH            Path to trained model file
  --input_path PATH            Path to image or directory
  --output_path PATH           Output directory

Optional:
  --patch_size INT             Patch size (default: 256)
  --overlap_ratio FLOAT        Patch overlap (default: 0.1)
  --save_confidence            Save confidence maps
  --save_visualizations        Save overlay visualizations
  --extensions LIST            Image extensions to process
```

### Evaluation Only

Evaluate a pre-trained model without training:

```bash
python main.py \
    --data_dir ./data \
    --evaluate_only \
    --model_path path/to/model.h5 \
    --dataset [train|val|test]
```

## Model Architecture

- **Base Architecture**: U-Net with encoder-decoder structure
- **Input**: 256×256×3 RGB image patches
- **Output**: 256×256×6 class probability maps
- **Parameters**: ~15.4 million parameters
- **Loss Function**: Combined Focal Loss + Dice Loss
- **Metrics**: Jaccard Coefficient, IoU, Precision, Recall

### Classes
1. **Building** (Purple)
2. **Land** (Light Purple)
3. **Road** (Light Blue)
4. **Vegetation** (Yellow)
5. **Water** (Orange)
6. **Unlabeled** (Gray)

## Training

### Data Pipeline
- **Patches**: 256×256 extracted from larger tiles
- **Augmentation**: Rotation, shifts, zoom, flips
- **Splits**: 70% train, 20% validation, 10% test
- **Normalization**: Pixel values scaled to [0,1]

### Training Process
1. **Patch Extraction**: Images divided into manageable patches
2. **Data Augmentation**: Applied during training
3. **Model Training**: U-Net trained with combined loss function
4. **Validation Monitoring**: Early stopping and checkpoint saving
5. **Artifact Generation**: Plots, metrics, and model files saved

### Outputs Generated
```
output_directory/
├── models/
│   └── unet_model.h5           # Final trained model
├── checkpoints/
│   └── Unet.weights.h5         # Best training weights
├── logs/
│   └── unet_timestamp/         # TensorBoard logs
├── plots/
│   └── unet_training_history.png
├── results/
│   └── unet_training_metrics.json
└── config.json
```

## Inference

### Processing Strategy
- **Large Images**: Processed using overlapping patches
- **Reconstruction**: Seamless stitching with weighted averaging
- **Memory Efficient**: Patch-by-patch processing prevents OOM errors
- **Quality Options**: Configurable overlap for speed vs quality trade-off

### Output Formats
1. **Class Predictions**: Colored segmentation masks (PNG)
2. **Confidence Maps**: Raw probability arrays (NPY)
3. **Overlay Visualizations**: Original + prediction composite (PNG)

### Batch Processing
```bash
# Process entire directory
python predict.py \
    --model_path model.h5 \
    --input_path aerial_images/ \
    --output_path results/ \
    --save_confidence \
    --save_visualizations
```

## Configuration

The project uses a centralized configuration system in `src/config/settings.py`:

### Key Settings
- **Patch Size**: 256×256 (should match training)
- **Classes**: 6 semantic classes
- **Data Splits**: 70/20/10 train/val/test
- **Augmentation**: Rotation, shifts, zoom, flips
- **Model Parameters**: Loss weights, metrics, callbacks

### Directory Structure
All outputs are organized into structured directories:
- `models/`: Final trained models
- `checkpoints/`: Training checkpoints
- `logs/`: TensorBoard logs
- `plots/`: Training visualizations
- `results/`: Evaluation metrics

## Results

### Training Performance
- **Epochs**: 200
- **Best Validation Jaccard**: 0.7879
- **Test Performance**: Accuracy: 0.8673, Jaccard Score: 0.7746

### Per-Class Performance (Example)
| Class       | F1 Score | Notes                                       |
|-------------|----------|---------------------------------------------|
| Land        | 0.9265   |Large homogeneous regions, easy to classify  |
| Water       | 0.8644   |Significant improvement from initial 0.06    |
| Building    | 0.8403   |Clear boundaries, distinct spectral signature|
| Road        | 0.7287   |Improved substantially from initial 0.006    |
| Vegetation  | 0.4552   |Linear features still challenging            |
| Unlabeled   | 0.0000   |Rare class, insufficient training examples   |


## License

MIT License - see LICENSE file for details.
