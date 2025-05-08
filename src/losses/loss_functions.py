import torch
import torch.nn as nn
import torch.nn.functional as F

class PrototypicalLoss(nn.Module):
    """
    Prototypical loss for few-shot learning
    
    This is the main classification loss that minimizes the distance between
    query examples and their true class prototypes.
    """
    def __init__(self):
        super(PrototypicalLoss, self).__init__()

    def forward(self, logits, target):
        """
        Args:
            logits: Negative distances to prototypes [batch_size, num_prototypes]
            target: Target classes [batch_size]
            
        Returns:
            loss: Loss value
        """
        return F.cross_entropy(logits, target)


class RefinementLoss(nn.Module):
    """
    Loss to ensure consistent prototype updates
    
    This loss encourages the refined prototypes to remain semantically
    close to the initial prototypes, preventing drastic changes.
    """
    def __init__(self, alpha=0.5):
        super(RefinementLoss, self).__init__()
        self.alpha = alpha

    def forward(self, prototypes, initial_prototypes):
        """
        Args:
            prototypes: Refined prototypes
            initial_prototypes: Initial prototypes
            
        Returns:
            loss: Loss value
        """
        # Compute cosine similarity between refined and initial prototypes
        cos_sim = F.cosine_similarity(prototypes, initial_prototypes, dim=1)

        # We want high similarity (close to 1), so we minimize 1 - similarity
        loss = torch.mean(1 - cos_sim)

        return self.alpha * loss


class IntraClassConsistencyLoss(nn.Module):
    """
    Loss to maintain coherence within disease classes
    
    This loss ensures that embeddings of the same class are consistent,
    helping to create more compact clusters in the embedding space.
    """
    def __init__(self, beta=0.3):
        super(IntraClassConsistencyLoss, self).__init__()
        self.beta = beta

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: Sample embeddings
            labels: Sample labels
            
        Returns:
            loss: Loss value
        """
        classes = torch.unique(labels)
        loss = 0.0

        for c in classes:
            # Get embeddings of class c
            mask = (labels == c)
            if torch.sum(mask) <= 1:
                continue

            class_embeddings = embeddings[mask]

            # Compute centroid
            centroid = torch.mean(class_embeddings, dim=0, keepdim=True)

            # Compute variance of distances to centroid
            distances = torch.sum((class_embeddings - centroid) ** 2, dim=1)
            variance = torch.var(distances)

            # Add to loss
            loss += variance

        return self.beta * loss / len(classes) if len(classes) > 0 else 0.0


class InterClassSeparationLoss(nn.Module):
    """
    Loss to push different disease prototypes apart
    
    This loss encourages separation between different class prototypes,
    improving discrimination between different diseases.
    """
    def __init__(self, gamma=0.2, margin=1.0):
        super(InterClassSeparationLoss, self).__init__()
        self.gamma = gamma
        self.margin = margin

    def forward(self, prototypes):
        """
        Args:
            prototypes: Class prototypes
            
        Returns:
            loss: Loss value
        """
        n_prototypes = prototypes.size(0)
        if n_prototypes <= 1:
            return 0.0

        # Compute pairwise distances between prototypes
        distances = torch.cdist(prototypes, prototypes, p=2)

        # Create a mask to select only the lower triangle (excluding diagonal)
        mask = torch.triu(torch.ones_like(distances), diagonal=1) == 1

        # Select distances using the mask
        distances = distances[mask]

        # Apply margin: we want distances to be at least margin
        loss = F.relu(self.margin - distances).mean()

        return self.gamma * loss


class SPROUTLoss(nn.Module):
    """
    Combined loss for SPROUT
    
    This combines the prototypical loss with the refinement, intra-class consistency,
    and inter-class separation losses to create a comprehensive training objective.
    """
    def __init__(self, alpha=0.5, beta=0.3, gamma=0.2, margin=1.0):
        super(SPROUTLoss, self).__init__()

        self.proto_loss = PrototypicalLoss()
        self.refine_loss = RefinementLoss(alpha=alpha)
        self.intra_loss = IntraClassConsistencyLoss(beta=beta)
        self.inter_loss = InterClassSeparationLoss(gamma=gamma, margin=margin)

    def forward(self, logits, targets, prototypes, initial_prototypes, support_embeddings, support_labels):
        """
        Combined loss computation
        
        Args:
            logits: Classification logits
            targets: Target labels
            prototypes: Refined prototypes
            initial_prototypes: Initial prototypes
            support_embeddings: Support set embeddings
            support_labels: Support set labels
            
        Returns:
            total_loss: Combined loss value
            loss_components: Dictionary of individual loss components
        """
        # Main classification loss
        proto_loss = self.proto_loss(logits, targets)

        # Prototype refinement loss
        refine_loss = self.refine_loss(prototypes, initial_prototypes)

        # Intra-class consistency loss
        intra_loss = self.intra_loss(support_embeddings, support_labels)

        # Inter-class separation loss
        inter_loss = self.inter_loss(prototypes)

        # Combine all losses
        total_loss = proto_loss + refine_loss + intra_loss + inter_loss

        # Return individual losses for monitoring
        return total_loss, {
            'proto_loss': proto_loss.item(),
            'refine_loss': refine_loss.item(),
            'intra_loss': intra_loss.item(),
            'inter_loss': inter_loss.item(),
            'total_loss': total_loss.item()
        }