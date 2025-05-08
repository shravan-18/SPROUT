# Dataset Directory

This directory is intended to contain the leaf disease datasets used for training and evaluating the SPROUT model.

## Expected Structure

```
data/
├── dataset_name/
│   ├── train/
│   │   ├── disease1/
│   │   │   ├── img1.jpg
│   │   │   ├── img2.jpg
│   │   │   └── ...
│   │   ├── disease2/
│   │   └── ...
│   └── test/
│       ├── disease1/
│       ├── disease2/
│       └── ...
```

## Supported Datasets

The code has been tested with the following plant leaf disease datasets:

1. Paddy (Rice) Leaf Disease Dataset
2. Cassava Leaf Disease Dataset
3. Sugarcane Leaf Disease Dataset

You can use other plant leaf disease datasets as long as they follow the directory structure outlined above.

## Data Preparation

1. Download the dataset of your choice
2. Extract and organize it into the train/test split structure
3. Place the dataset folder in this directory

Note: Large datasets should not be committed to GitHub. Consider using Git LFS or keeping the data in a separate location referenced by path during training.