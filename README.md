# SPROUT: Symptom-centric Prototypical Representation Optimization and Uncertainty-aware Tuning for Few-Shot Precision Agriculture

This repository contains the official implementation of SPROUT, a novel few-shot learning approach for plant leaf disease classification.

## Overview

SPROUT (Symptom-centric Prototypical Representation Optimization and Uncertainty-aware Tuning) is designed to quickly adapt to new plant diseases with minimal examples. The method combines prototypical networks with an attention-based prototype refinement mechanism that focuses on disease-relevant features.

## Key Features

- **Disease-Aware Prototype Refinement**: Optimizes prototypes specifically for plant disease characteristics
- **Hierarchical Prototype Representation**: Represents diseases at multiple granularity levels
- **Confidence-Guided Adaptation**: Implements uncertainty-aware few-shot learning
- **Symptom-Based Similarity Metrics**: Creates custom similarity metrics that emphasize disease-relevant features

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/sprout.git
cd sprout

# Install dependencies
pip install -r requirements.txt
```

## Dataset Structure

The dataset should be organized in the following structure:

```
dataset_root/
├── train/
│   ├── disease1/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   ├── disease2/
│   └── ...
└── test/
    ├── disease1/
    ├── disease2/
    └── ...
```

## Usage 

```
python experiments/train.py \
    --data_dir /path/to/dataset \
    --output_dir ./results \
    --backbone resnet50 \
    --n_way 5 \
    --k_shot 5 \
    --n_query 15 \
    --n_episodes 100 \
    --num_epochs 10
```

## Evaluation
```bash
python experiments/evaluate.py \
    --model_path ./results/sprout_model_final.pth \
    --data_dir /path/to/dataset \
    --output_dir ./evaluation_results \
    --n_way 5 \
    --k_shot 1 5 10 \
    --n_query 15 \
    --n_episodes 100
```

## Comparison with Standard Models
```bash
python experiments/compare_models.py \
    --model_path ./results/sprout_model_final.pth \
    --data_dir /path/to/dataset \
    --output_dir ./comparison_results \
    --n_way 5 \
    --k_shot 1 5 10 \
    --n_query 15 \
    --n_episodes 50
```

## Model Architecture

**SPROUT** consists of the following components:

1. **Feature Extractor**: Uses a pre-trained CNN backbone (ResNet50 or EfficientNet) to extract high-level visual features.
2. **Embedding Network**: Projects features to a lower-dimensional embedding space where semantic relationships between disease symptoms are preserved.
3. **Prototype Module**: Generates and refines class prototypes using an attention mechanism that focuses on disease-relevant features.
4. **Classification Head**: Makes predictions based on distances to refined prototypes.
