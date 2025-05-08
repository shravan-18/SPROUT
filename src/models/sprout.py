import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.feature_extractor import FeatureExtractor
from src.models.embedding_network import EmbeddingNetwork
from src.models.prototype_module import PrototypeModule

class SPROUT(nn.Module):
    """
    SPROUT: Symptom-centric Prototypical Representation Optimization and Uncertainty-aware Tuning
    
    This is the main model that integrates the feature extractor, embedding network,
    and prototype module for few-shot learning of plant leaf diseases.
    
    The model can operate in two modes:
    1. Few-shot mode: Uses support images to create prototypes for classification
    2. Feature extraction mode: Extracts embeddings for visualization or analysis
    """
    def __init__(self, num_classes, backbone='resnet50', embed_dim=128,
                 hidden_dims=None, num_refinement_steps=3, temperature=10.0):
        """
        Args:
            num_classes: Number of classes in the dataset
            backbone: Feature extractor backbone ('resnet50' or 'efficientnet_b0')
            embed_dim: Dimension of the embedding space
            hidden_dims: List of hidden layer dimensions for the embedding network
            num_refinement_steps: Number of prototype refinement iterations
            temperature: Temperature scaling factor for distance calculations
        """
        super(SPROUT, self).__init__()

        # Feature extractor
        self.feature_extractor = FeatureExtractor(backbone=backbone)
        feature_dim = self.feature_extractor.get_feature_dim()

        # Embedding network
        self.embedding_network = EmbeddingNetwork(
            input_dim=feature_dim,
            embed_dim=embed_dim,
            hidden_dims=hidden_dims
        )

        # Prototype module
        self.prototype_module = PrototypeModule(
            embed_dim=embed_dim,
            num_refinement_steps=num_refinement_steps
        )

        # Classification parameters
        self.num_classes = num_classes
        self.temperature = temperature

    def forward(self, query_images, support_images=None, support_labels=None):
        """
        Forward pass with dual-mode functionality
        
        Args:
            query_images: Query images to classify
            support_images: Support set images (optional, for few-shot)
            support_labels: Support set labels (optional, for few-shot)
            
        Returns:
            In few-shot mode: (logits, prototypes)
            In feature extraction mode: query_embeddings
        """
        # Extract features and compute embeddings for query images
        query_features = self.feature_extractor(query_images)
        query_embeddings = self.embedding_network(query_features)

        # If in few-shot mode (support set provided)
        if support_images is not None and support_labels is not None:
            # Extract features and compute embeddings for support images
            support_features = self.feature_extractor(support_images)
            support_embeddings = self.embedding_network(support_features)

            # Generate and refine prototypes
            prototypes = self.prototype_module(support_embeddings, support_labels)

            # Compute distances to prototypes
            # Using negative squared Euclidean distance
            logits = -self.compute_distances(query_embeddings, prototypes)

            return logits, prototypes
        else:
            # Feature extraction mode - just return embeddings
            return query_embeddings

    def compute_distances(self, embeddings, prototypes):
        """
        Compute squared Euclidean distances between embeddings and prototypes
        
        Args:
            embeddings: Query embeddings [batch_size, embed_dim]
            prototypes: Class prototypes [num_classes, embed_dim]
            
        Returns:
            distances: Distance matrix [batch_size, num_classes]
        """
        # Expand dimensions for broadcasting
        embeddings_expanded = embeddings.unsqueeze(1)  # [batch_size, 1, embed_dim]
        prototypes_expanded = prototypes.unsqueeze(0)  # [1, num_classes, embed_dim]

        # Compute squared Euclidean distance
        distances = torch.sum((embeddings_expanded - prototypes_expanded) ** 2, dim=2)

        # Apply temperature scaling
        distances = distances / self.temperature

        return distances

    def extract_features(self, images):
        """
        Helper method to extract features for visualization or analysis
        
        Args:
            images: Input images
            
        Returns:
            embeddings: Embeddings in the metric space
        """
        features = self.feature_extractor(images)
        embeddings = self.embedding_network(features)
        return embeddings