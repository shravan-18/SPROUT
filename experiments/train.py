import os
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

from src.models.sprout import SPROUT
from src.losses.loss_functions import SPROUTLoss
from src.data.dataset import load_dataset
from src.utils.episode_sampler import EpisodeBatchSampler, episode_collate
from src.utils.visualization import plot_training_curves

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_episode_batch(model, optimizer, criterion, episode_batch, device, n_way, k_shot, n_query):
    """
    Train on a batch of episodes with error handling
    """
    # Unpack the batch
    images, labels = episode_batch
    images = images.to(device)
    labels = labels.to(device)

    # Reshape for episodic training
    # We need to organize the batch into a proper n-way k-shot episode

    # First, create a mapping from original labels to episode labels (0 to n_way-1)
    unique_labels = torch.unique(labels)
    if len(unique_labels) != n_way:
        # Handle case where batch doesn't have exactly n_way classes
        print(f"Warning: Expected {n_way} classes but got {len(unique_labels)}")
        n_way = len(unique_labels)

    if n_way < 2:
        # Not enough classes to train
        print("Warning: Need at least 2 classes for a meaningful episode")
        return 0.0, {'total_loss': 0.0, 'proto_loss': 0.0, 'refine_loss': 0.0,
                    'intra_loss': 0.0, 'inter_loss': 0.0}

    label_mapping = {unique_labels[i].item(): i for i in range(n_way)}

    # Convert original labels to episode labels
    episode_labels = torch.tensor([label_mapping[label.item()] for label in labels],
                                 device=device)

    # Organize support and query sets
    support_indices = []
    query_indices = []

    for i in range(n_way):
        # Find indices for this class
        class_indices = torch.nonzero(episode_labels == i, as_tuple=True)[0]

        if len(class_indices) == 0:
            continue

        if len(class_indices) < k_shot + 1:  # Need at least 1 query
            # If we don't have enough samples, use what we have
            support_count = min(k_shot, len(class_indices) - 1)
            if support_count <= 0:
                continue

            support_indices.append(class_indices[:support_count])
            query_indices.append(class_indices[support_count:])
        else:
            # Normal case - enough samples
            support_indices.append(class_indices[:k_shot])
            query_count = min(n_query, len(class_indices) - k_shot)
            query_indices.append(class_indices[k_shot:k_shot+query_count])

    # Make sure we have support and query samples
    if not support_indices or not query_indices:
        print("Warning: Not enough support or query samples")
        return 0.0, {'total_loss': 0.0, 'proto_loss': 0.0, 'refine_loss': 0.0,
                    'intra_loss': 0.0, 'inter_loss': 0.0}

    # Flatten the indices
    support_indices = torch.cat(support_indices)
    query_indices = torch.cat(query_indices)

    if len(support_indices) == 0 or len(query_indices) == 0:
        print("Warning: Empty support or query set after processing")
        return 0.0, {'total_loss': 0.0, 'proto_loss': 0.0, 'refine_loss': 0.0,
                    'intra_loss': 0.0, 'inter_loss': 0.0}

    # Extract the support and query data
    support_images = images[support_indices]
    support_labels = episode_labels[support_indices]
    query_images = images[query_indices]
    query_labels = episode_labels[query_indices]

    # Check if we have all classes in support set
    support_classes = torch.unique(support_labels)
    if len(support_classes) < 2:
        # Not enough classes in support set
        print("Warning: Not enough classes in support set")
        return 0.0, {'total_loss': 0.0, 'proto_loss': 0.0, 'refine_loss': 0.0,
                    'intra_loss': 0.0, 'inter_loss': 0.0}

    # Train on this episode
    try:
        model.train()
        optimizer.zero_grad()

        # Get support embeddings
        support_features = model.feature_extractor(support_images)
        support_embeddings = model.embedding_network(support_features)

        # Generate initial prototypes
        initial_prototypes = model.prototype_module.generate_initial_prototypes(support_embeddings, support_labels)

        # Forward pass
        logits, prototypes = model(query_images, support_images, support_labels)

        # Compute loss
        loss, loss_components = criterion(logits, query_labels, prototypes, initial_prototypes,
                                         support_embeddings, support_labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Compute accuracy
        _, predicted = torch.max(logits.data, 1)
        accuracy = (predicted == query_labels).float().mean().item()

        return accuracy, loss_components

    except Exception as e:
        print(f"Error in training episode: {e}")
        return 0.0, {'total_loss': 0.0, 'proto_loss': 0.0, 'refine_loss': 0.0,
                    'intra_loss': 0.0, 'inter_loss': 0.0}


def train_model(model, train_dataset, criterion, optimizer, scheduler=None,
               n_way=5, k_shot=5, n_query=15, n_episodes=100, num_epochs=10,
               batch_size=4, eval_interval=10, device=torch.device("cpu"),
               output_dir="./results"):
    """
    Train the SPROUT model with episode-based few-shot learning
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Move model to device
    model = model.to(device)

    # Create episode batch sampler
    sampler = EpisodeBatchSampler(train_dataset, n_way, k_shot, n_query, n_episodes)

    # Create dataloader
    dataloader = DataLoader(
        train_dataset, batch_sampler=sampler, num_workers=4, collate_fn=episode_collate
    )

    # Track metrics
    train_accuracies = []
    train_losses = []

    # Train for specified number of epochs
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # Train on episodes
        episode_accuracies = []
        episode_losses = []

        for i, episode_batch in enumerate(tqdm(dataloader, desc="Episodes")):
            # Train on this episode batch
            accuracy, loss_components = train_episode_batch(
                model, optimizer, criterion, episode_batch, device, n_way, k_shot, n_query
            )

            # Track metrics
            episode_accuracies.append(accuracy)
            episode_losses.append(loss_components['total_loss'])

            # Print progress
            if (i+1) % eval_interval == 0:
                print(f"Episode {i+1}/{n_episodes}, " +
                      f"Accuracy: {np.mean(episode_accuracies[-eval_interval:]):.4f}, " +
                      f"Loss: {np.mean(episode_losses[-eval_interval:]):.4f}")

        # Update learning rate
        if scheduler is not None:
            scheduler.step()

        # Compute epoch metrics
        epoch_accuracy = np.mean(episode_accuracies)
        epoch_loss = np.mean(episode_losses)

        train_accuracies.append(epoch_accuracy)
        train_losses.append(epoch_loss)

        print(f"Epoch {epoch+1} summary - Accuracy: {epoch_accuracy:.4f}, Loss: {epoch_loss:.4f}")
        
        # Save model checkpoint
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            checkpoint_path = os.path.join(output_dir, f'sprout_model_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model checkpoint saved to {checkpoint_path}")

    # Save final model
    final_model_path = os.path.join(output_dir, 'sprout_model_final.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")

    # Plot training curves
    plot_training_curves(train_accuracies, train_losses)
    
    return train_accuracies, train_losses


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train SPROUT model for few-shot leaf disease classification')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory for model and results')
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['resnet50', 'efficientnet_b0'], 
                        help='Feature extractor backbone')
    parser.add_argument('--embed_dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--n_way', type=int, default=5, help='Number of classes per episode')
    parser.add_argument('--k_shot', type=int, default=5, help='Number of support examples per class')
    parser.add_argument('--n_query', type=int, default=15, help='Number of query examples per class')
    parser.add_argument('--n_episodes', type=int, default=100, help='Number of episodes per epoch')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    args = parser.parse_args()

    # Set random seed
    set_seed(args.seed)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    train_dataset, _ = load_dataset(args.data_dir)
    num_classes = len(train_dataset.get_classes())

    # Create model
    model = SPROUT(
        num_classes=num_classes,
        backbone=args.backbone,
        embed_dim=args.embed_dim,
        hidden_dims=[512, 256],
        num_refinement_steps=3
    )

    # Create criterion
    criterion = SPROUTLoss(alpha=0.5, beta=0.3, gamma=0.2, margin=1.0)

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Create scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # Train model
    print("Training SPROUT model...")
    train_accuracies, train_losses = train_model(
        model=model,
        train_dataset=train_dataset,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        n_way=args.n_way,
        k_shot=args.k_shot,
        n_query=args.n_query,
        n_episodes=args.n_episodes,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        eval_interval=10,
        device=device,
        output_dir=args.output_dir
    )

    print(f"Training completed. Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()