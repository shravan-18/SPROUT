import torch
import torch.nn as nn
from torchvision import models

class FeatureExtractor(nn.Module):
    """
    CNN backbone for feature extraction using a pre-trained model
    
    This module serves as the feature extraction component of the SPROUT model,
    using a pre-trained backbone network (ResNet50, EfficientNet) to extract 
    high-level visual features from leaf images.
    """
    def __init__(self, backbone='resnet50', pretrained=True, freeze_backbone=False):
        super(FeatureExtractor, self).__init__()

        # Load pre-trained model
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            self.feature_dim = self.backbone.fc.in_features
            # Remove the final FC layer
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        elif backbone == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            self.feature_dim = self.backbone.classifier[1].in_features
            # Remove the classifier
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        """
        Forward pass through the feature extractor
        
        Args:
            x: Input images of shape [batch_size, 3, height, width]
            
        Returns:
            features: Extracted features of shape [batch_size, feature_dim]
        """
        features = self.backbone(x)
        # Flatten the features
        features = features.view(features.size(0), -1)
        return features

    def get_feature_dim(self):
        """Return the feature dimension of the backbone"""
        return self.feature_dim