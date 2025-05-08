import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path

class LeafDiseaseDataset(Dataset):
    """
    Dataset class for loading leaf disease images for SPROUT
    
    This dataset is designed to work with various plant leaf disease datasets
    organized in a folder structure where each disease class has its own folder.
    """
    def __init__(self, root_dir, transform=None, split='train', use_cache=False):
        """
        Args:
            root_dir (string): Directory with train and test folders
            transform (callable, optional): Transform to be applied on images
            split (string): 'train' or 'test'
            use_cache (bool): Whether to cache images in memory
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.split = split
        self.use_cache = use_cache

        # Path to split folder (train or test)
        self.split_dir = self.root_dir / split

        # Get all class folders
        self.classes = sorted([d.name for d in self.split_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        # Get all image paths and their labels
        self.image_paths = []
        self.labels = []

        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.JPG', '*.JPEG', '*.PNG']
        for class_name in self.classes:
            class_dir = self.split_dir / class_name
            for ext in image_extensions:
                for img_path in class_dir.glob(f'**/{ext}'):
                    self.image_paths.append(img_path)
                    self.labels.append(self.class_to_idx[class_name])

        # Convert labels to tensor
        self.labels = torch.tensor(self.labels)

        # Pre-compute indices by class for efficient sampling
        self.indices_by_class = {}
        for class_idx in range(len(self.classes)):
            self.indices_by_class[class_idx] = torch.where(self.labels == class_idx)[0].tolist()

        # Create feature cache if enabled
        self.feature_cache = {} if use_cache else None

        print(f"Found {len(self.image_paths)} images across {len(self.classes)} classes in {split} set")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Check cache first if enabled
        if self.use_cache and idx in self.feature_cache:
            return self.feature_cache[idx], label

        # Load and transform image
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a placeholder image in case of error
            image = torch.zeros((3, 224, 224)) if self.transform else Image.new('RGB', (224, 224))

        # Cache if enabled
        if self.use_cache:
            self.feature_cache[idx] = image

        return image, label

    def get_classes(self):
        """Return list of class names"""
        return self.classes

    def get_class_counts(self):
        """Return count of images in each class"""
        class_counts = {}
        for i, cls in enumerate(self.classes):
            class_counts[cls] = len(self.indices_by_class[i])
        return class_counts

    def get_indices_by_class(self):
        """Return dict mapping class indices to image indices"""
        return self.indices_by_class


def get_transforms():
    """
    Returns transforms for training and testing
    """
    from torchvision import transforms
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, test_transform


def load_dataset(root_dir):
    """
    Load the leaf disease dataset with appropriate transforms
    """
    train_transform, test_transform = get_transforms()

    train_dataset = LeafDiseaseDataset(root_dir, transform=train_transform, split='train')
    test_dataset = LeafDiseaseDataset(root_dir, transform=test_transform, split='test')

    print(f"Train dataset: {len(train_dataset)} images across {len(train_dataset.get_classes())} classes")
    print(f"Test dataset: {len(test_dataset)} images across {len(test_dataset.get_classes())} classes")

    # Print class distribution
    train_counts = train_dataset.get_class_counts()
    print("\nClass distribution in training set:")
    for cls, count in train_counts.items():
        print(f"{cls}: {count} images")

    return train_dataset, test_dataset
    