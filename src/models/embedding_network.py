import torch
import torch.nn as nn
import torch.nn.functional as F

class EmbeddingNetwork(nn.Module):
    """
    Embedding network that projects features to a metric space
    
    This module maps the high-dimensional features from the feature extractor
    into a lower-dimensional embedding space where semantic relationships
    between disease symptoms are preserved and can be compared using
    distance metrics.
    """
    def __init__(self, input_dim, embed_dim=128, hidden_dims=None):
        """
        Args:
            input_dim: Dimension of input features
            embed_dim: Dimension of output embeddings
            hidden_dims: List of hidden layer dimensions (if None, defaults to [512, 256])
        """
        super(EmbeddingNetwork, self).__init__()

        if hidden_dims is None:
            hidden_dims = [512, 256]

        # Build layers
        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.ReLU(inplace=True))

        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.BatchNorm1d(hidden_dims[i+1]))
            layers.append(nn.ReLU(inplace=True))

        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], embed_dim))

        self.embedding_layers = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the embedding network
        
        Args:
            x: Input features of shape [batch_size, input_dim]
            
        Returns:
            embeddings: L2 normalized embeddings of shape [batch_size, embed_dim]
        """
        embeddings = self.embedding_layers(x)
        # L2 normalize the embeddings for cosine similarity calculations
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings