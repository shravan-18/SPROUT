import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt

from src.models.sprout import SPROUT
from src.data.dataset import load_dataset
from src.utils.episode_sampler import EpisodeBatchSampler, episode_collate
from src.utils.visualization import (
    plot_embeddings_tsne, 
    plot_confusion_matrix, 
    visualize_distance_heatmap,
    plot_shot_comparison
)

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate_episode(model, support_images, support_labels, query_images, query_labels, device):
    """
    Evaluate on a single episode
    """
    # Move data to device
    support_images = support_images.to(device)
    support_labels = support_labels.to(device)
    query_images = query_images.to(device)
    query_labels = query_labels.to(device)

    # Set model to evaluation mode
    model.eval()

    # Forward pass
    with torch.no_grad():
        logits, prototypes = model(query_images, support_images, support_labels)

        # Compute predictions
        _, predicted = torch.max(logits.data, 1)

        # Compute accuracy
        accuracy = (predicted == query_labels).float().mean().item()

        # Compute embeddings for visualization
        query_features = model.feature_extractor(query_images)
        query_embeddings = model.embedding_network(query_features)

        support_features = model.feature_extractor(support_images)
        support_embeddings = model.embedding_network(support_features)

    # Convert everything to numpy for easier handling
    query_embeddings_np = query_embeddings.cpu().numpy()
    query_labels_np = query_labels.cpu().numpy()
    query_pred_np = predicted.cpu().numpy()

    support_embeddings_np = support_embeddings.cpu().numpy()
    support_labels_np = support_labels.cpu().numpy()

    prototypes_np = prototypes.cpu().numpy()

    return {
        'accuracy': accuracy,
        'query_embeddings': query_embeddings_np,
        'query_labels': query_labels_np,
        'query_predictions': query_pred_np,
        'support_embeddings': support_embeddings_np,
        'support_labels': support_labels_np,
        'prototypes': prototypes_np
    }


def evaluate_model(model, test_dataset, n_way=5, k_shot=5, n_query=15, n_episodes=100, device=torch.device("cpu")):
    """
    Evaluate the model on few-shot tasks
    """
    # Move model to device
    model = model.to(device)

    # Set model to evaluation mode
    model.eval()

    # Create episode sampler
    sampler = EpisodeBatchSampler(
        test_dataset, n_way=n_way, k_shot=k_shot,
        n_query=n_query, n_episodes=n_episodes
    )

    # Create dataloader
    dataloader = DataLoader(
        test_dataset, batch_sampler=sampler, num_workers=4, collate_fn=episode_collate
    )

    # Evaluate on episodes
    episode_accuracies = []
    all_predictions = []
    all_true_labels = []

    for i, batch in enumerate(tqdm(dataloader, desc="Evaluating", total=n_episodes)):
        # Process the batch into an episode
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)

        # Get unique classes in this batch
        unique_classes = torch.unique(labels)
        if len(unique_classes) < 2:
            # Skip episodes with less than 2 classes
            continue

        # Create mapping from original labels to episode labels (0 to n_way-1)
        label_mapping = {unique_classes[i].item(): i for i in range(min(n_way, len(unique_classes)))}

        # Convert to episode labels
        episode_labels = torch.tensor([label_mapping.get(l.item(), 0)
                                      for l in labels if l.item() in label_mapping],
                                     device=device)

        # Create indices for support and query
        support_indices = []
        query_indices = []

        for i in range(len(label_mapping)):
            # Find indices for this class
            class_indices = torch.nonzero(episode_labels == i, as_tuple=True)[0]

            if len(class_indices) < k_shot + 1:
                # Skip classes with not enough samples
                continue

            support_indices.extend(class_indices[:k_shot].tolist())
            query_indices.extend(class_indices[k_shot:k_shot+min(n_query, len(class_indices)-k_shot)].tolist())

        if not support_indices or not query_indices:
            # Skip episodes without enough samples
            continue

        # Get support and query sets
        support_images = images[support_indices]
        support_labels = episode_labels[support_indices]
        query_images = images[query_indices]
        query_labels = episode_labels[query_indices]

        # Evaluate
        with torch.no_grad():
            logits, _ = model(query_images, support_images, support_labels)

            # Compute predictions
            _, predicted = torch.max(logits.data, 1)

            # Compute accuracy
            accuracy = (predicted == query_labels).float().mean().item()
            episode_accuracies.append(accuracy)

            # Store predictions and true labels for metrics
            all_predictions.extend(predicted.cpu().numpy())
            all_true_labels.extend(query_labels.cpu().numpy())

        # Print progress
        if (i+1) % 10 == 0:
            print(f"Episode {i+1}/{n_episodes}, " +
                  f"Accuracy: {np.mean(episode_accuracies[-10:]):.4f}")

    # Compute overall metrics
    if episode_accuracies:
        overall_accuracy = np.mean(episode_accuracies)
        print(f"Overall accuracy: {overall_accuracy:.4f}")
    else:
        overall_accuracy = 0.0
        print("Warning: No valid episodes for evaluation")

    return overall_accuracy, (all_true_labels, all_predictions)


def evaluate_across_shots(model, test_dataset, n_way=5, shot_values=[1, 5, 10],
                          n_query=15, n_episodes=100, device=torch.device("cpu")):
    """
    Evaluate the model across different shot values
    """
    results = {}

    for k_shot in shot_values:
        print(f"\nEvaluating {n_way}-way {k_shot}-shot:")
        accuracy, _ = evaluate_model(
            model, test_dataset, n_way=n_way, k_shot=k_shot,
            n_query=n_query, n_episodes=n_episodes, device=device
        )
        results[k_shot] = accuracy

    return results


def generate_visualizations(model, test_dataset, output_dir, n_way=5, k_shot=5, n_query=15, device=torch.device("cpu")):
    """
    Generate visualizations for analysis
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a sampler for a single episode
    from src.utils.episode_sampler import EpisodeSampler
    episode_sampler = EpisodeSampler(
        test_dataset, n_way=n_way, k_shot=k_shot, n_query=n_query, n_episodes=1
    )
    
    # Get a single episode
    support_indices, query_indices, episode_classes = next(iter(episode_sampler))
    
    # Create episode batch
    from src.utils.episode_sampler import create_episode_batch
    support_images, support_labels, query_images, query_labels = create_episode_batch(
        test_dataset, support_indices, query_indices, episode_classes
    )
    
    # Evaluate on this episode
    results = evaluate_episode(
        model, support_images, support_labels, query_images, query_labels, device
    )
    
    # 1. Embeddings visualization
    all_embeddings = np.vstack([results['support_embeddings'], results['query_embeddings']])
    all_labels = np.concatenate([results['support_labels'], results['query_labels']])
    
    plt.figure(figsize=(10, 8))
    plot_embeddings_tsne(all_embeddings, all_labels, results['prototypes'],
                      title=f"{n_way}-way {k_shot}-shot Embedding Space")
    plt.savefig(os.path.join(output_dir, f"embedding_space_{n_way}way_{k_shot}shot.png"), dpi=300)
    plt.close()
    
    # 2. Confusion matrix
    plt.figure(figsize=(10, 8))
    plot_confusion_matrix(results['query_labels'], results['query_predictions'],
                      title=f"{n_way}-way {k_shot}-shot Confusion Matrix")
    plt.savefig(os.path.join(output_dir, f"confusion_matrix_{n_way}way_{k_shot}shot.png"), dpi=300)
    plt.close()
    
    # 3. Distance heatmap
    plt.figure(figsize=(10, 8))
    visualize_distance_heatmap(results['query_embeddings'], results['prototypes'],
                            results['query_labels'])
    plt.savefig(os.path.join(output_dir, f"distance_heatmap_{n_way}way_{k_shot}shot.png"), dpi=300)
    plt.close()
    
    print(f"Visualizations saved to {output_dir}")
    
    return results


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate SPROUT model on few-shot tasks')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results', help='Output directory for results')
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['resnet50', 'efficientnet_b0'], 
                        help='Feature extractor backbone')
    parser.add_argument('--embed_dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--n_way', type=int, default=5, help='Number of classes per episode')
    parser.add_argument('--k_shot', type=int, nargs='+', default=[1, 5, 10], help='Number of support examples per class')
    parser.add_argument('--n_query', type=int, default=15, help='Number of query examples per class')
    parser.add_argument('--n_episodes', type=int, default=100, help='Number of episodes to evaluate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    args = parser.parse_args()

    # Set random seed
    set_seed(args.seed)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    _, test_dataset = load_dataset(args.data_dir)
    num_classes = len(test_dataset.get_classes())

    # Create model
    model = SPROUT(
        num_classes=num_classes,
        backbone=args.backbone,
        embed_dim=args.embed_dim,
        hidden_dims=[512, 256],
        num_refinement_steps=3
    )

    # Load trained weights
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Evaluate across different shot values
    print("Evaluating model across different shot values...")
    shot_results = evaluate_across_shots(
        model, test_dataset, n_way=args.n_way, shot_values=args.k_shot,
        n_query=args.n_query, n_episodes=args.n_episodes, device=device
    )

    # Plot shot comparison
    plt.figure(figsize=(10, 6))
    plot_shot_comparison(shot_results)
    plt.savefig(os.path.join(args.output_dir, f"shot_comparison_{args.n_way}way.png"), dpi=300)
    plt.close()
    
    # Save results to file
    with open(os.path.join(args.output_dir, 'shot_results.txt'), 'w') as f:
        f.write(f"SPROUT Model Evaluation Results ({args.n_way}-way classification)\n")
        f.write("-" * 50 + "\n")
        for k, acc in shot_results.items():
            f.write(f"{k}-shot: {acc:.4f}\n")
    
    # Generate visualizations for the middle shot value
    middle_shot = args.k_shot[len(args.k_shot) // 2]
    vis_dir = os.path.join(args.output_dir, 'visualizations')
    generate_visualizations(
        model, test_dataset, vis_dir, n_way=args.n_way, 
        k_shot=middle_shot, n_query=args.n_query, device=device
    )

    print(f"Evaluation completed. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()