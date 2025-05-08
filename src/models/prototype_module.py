import torch
import torch.nn as nn
import torch.nn.functional as F

class PrototypeModule(nn.Module):
    """
    Module for generating and refining prototypes
    
    This module implements the Symptom-centric Prototypical Representation Optimization
    component of SPROUT. It creates initial prototypes by averaging support embeddings
    and then refines them through an attention mechanism that focuses on disease-relevant
    features.
    """
    def __init__(self, embed_dim=128, num_refinement_steps=3):
        """
        Args:
            embed_dim: Dimension of the embedding space
            num_refinement_steps: Number of prototype refinement iterations
        """
        super(PrototypeModule, self).__init__()

        self.embed_dim = embed_dim
        self.num_refinement_steps = num_refinement_steps

        # Attention mechanism for prototype refinement
        self.attention = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 2, 1)
        )

        # Prototype refinement network
        self.refinement_network = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),  # Takes concatenated [prototype, weighted_avg]
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim)
        )

    def generate_initial_prototypes(self, support_embeddings, support_labels):
        """
        Generate initial prototypes by averaging embeddings of each class
        
        Args:
            support_embeddings: Embeddings of support set [N*K, embed_dim]
            support_labels: Labels of support set [N*K]
            
        Returns:
            prototypes: Initial prototypes [N, embed_dim]
        """
        classes = torch.unique(support_labels)
        prototypes = []

        for c in classes:
            # Get embeddings of class c
            class_mask = (support_labels == c)
            class_embeddings = support_embeddings[class_mask]

            if len(class_embeddings) == 0:
                # Handle empty class case
                prototype = torch.zeros(support_embeddings.size(1), device=support_embeddings.device)
            else:
                # Average the embeddings to get the prototype
                prototype = torch.mean(class_embeddings, dim=0)

            prototypes.append(prototype)

        return torch.stack(prototypes)

    def compute_attention_weights(self, support_embeddings, prototype, support_labels, class_idx):
        """
        Compute attention weights for support samples based on their relevance to the prototype
        
        Args:
            support_embeddings: Embeddings of support set
            prototype: Current prototype for the class
            support_labels: Labels of support set
            class_idx: Class index to calculate attention for
            
        Returns:
            attention_weights: Weights for each support sample of the class
            class_embeddings: Embeddings of samples from the class
        """
        # Get embeddings of class class_idx
        class_mask = (support_labels == class_idx)
        class_embeddings = support_embeddings[class_mask]

        if len(class_embeddings) == 0:
            return None

        # Expand prototype to match class_embeddings size
        prototype_expanded = prototype.unsqueeze(0).expand(class_embeddings.size(0), -1)

        # Concatenate class embeddings with prototype for attention
        attention_input = torch.cat([class_embeddings, prototype_expanded], dim=1)

        # Compute attention scores
        attention_scores = self.attention(attention_input)

        # Convert to weights using softmax
        attention_weights = F.softmax(attention_scores, dim=0)

        return attention_weights, class_embeddings

    def refine_prototype(self, prototype, support_embeddings, support_labels, class_idx):
        """
        Refine a prototype using attention mechanism
        
        Args:
            prototype: Current prototype for the class
            support_embeddings: Embeddings of support set
            support_labels: Labels of support set
            class_idx: Class index to refine prototype for
            
        Returns:
            refined_prototype: Refined prototype for the class
        """
        attention_result = self.compute_attention_weights(
            support_embeddings, prototype, support_labels, class_idx)

        if attention_result is None:
            return prototype

        attention_weights, class_embeddings = attention_result

        # Weighted average of support embeddings
        weighted_avg = torch.sum(attention_weights * class_embeddings, dim=0)

        # Concatenate original prototype with weighted average
        refinement_input = torch.cat([prototype, weighted_avg], dim=0)

        # Refine the prototype
        refined_prototype = self.refinement_network(refinement_input.unsqueeze(0)).squeeze(0)

        # Normalize the refined prototype
        refined_prototype = F.normalize(refined_prototype, p=2, dim=0)

        return refined_prototype

    def forward(self, support_embeddings, support_labels):
        """
        Generate and refine prototypes
        
        Args:
            support_embeddings: Embeddings of support set [N*K, embed_dim]
            support_labels: Labels of support set [N*K]
            
        Returns:
            refined_prototypes: Refined prototypes [N, embed_dim]
        """
        # Handle case of no support examples
        if len(support_embeddings) == 0:
            return torch.zeros((0, self.embed_dim), device=support_embeddings.device)

        # Generate initial prototypes
        prototypes = self.generate_initial_prototypes(support_embeddings, support_labels)

        # Get unique classes
        classes = torch.unique(support_labels)
        if len(classes) == 0:
            return prototypes

        # Refine prototypes through multiple iterations
        for _ in range(self.num_refinement_steps):
            refined_prototypes = []

            for i, c in enumerate(classes):
                if i >= len(prototypes):
                    break

                refined_prototype = self.refine_prototype(
                    prototypes[i], support_embeddings, support_labels, c)
                refined_prototypes.append(refined_prototype)

            prototypes = torch.stack(refined_prototypes) if refined_prototypes else prototypes

        return prototypes