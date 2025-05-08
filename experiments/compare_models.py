import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models
import timm
from tqdm import tqdm
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score, matthews_corrcoef

from src.models.sprout import SPROUT
from src.data.dataset import load_dataset
from src.utils.episode_sampler import EpisodeBatchSampler, episode_collate

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate_standard_model_few_shot(model_base, test_dataset, n_way=5, k_shot=5, n_query=15,
                                    n_episodes=50, device=torch.device("cuda")):
    """
    Evaluate a standard model in a few-shot learning setting
    Using the model as a feature extractor + nearest centroid classifier
    """
    # Create episode sampler
    sampler = EpisodeBatchSampler(
        test_dataset, n_way=n_way, k_shot=k_shot+n_query, n_query=0, n_episodes=n_episodes
    )

    # Create dataloader
    dataloader = DataLoader(
        test_dataset, batch_sampler=sampler, num_workers=4, collate_fn=episode_collate
    )

    episode_accuracies = []
    all_predictions = []
    all_true_labels = []

    # Use model as feature extractor + nearest centroid
    for batch in tqdm(dataloader, desc=f"Few-shot evaluation"):
        images, labels = batch
        images = images.to(device)

        # Get unique classes
        unique_classes = torch.unique(labels)
        if len(unique_classes) < n_way:
            continue

        # Create mapping from original labels to episode labels
        label_mapping = {unique_classes[i].item(): i for i in range(n_way)}
        episode_labels = torch.tensor([label_mapping[l.item()] for l in labels if l.item() in label_mapping],
                                     device=device)

        # Split into support and query
        support_indices = []
        query_indices = []

        for cls_idx in range(n_way):
            cls_indices = torch.nonzero(episode_labels == cls_idx, as_tuple=True)[0]
            if len(cls_indices) < k_shot + 1:
                continue

            support_indices.extend(cls_indices[:k_shot].tolist())
            query_indices.extend(cls_indices[k_shot:k_shot+min(n_query, len(cls_indices)-k_shot)].tolist())

        if len(support_indices) < n_way*k_shot or len(query_indices) == 0:
            continue

        support_images = images[support_indices]
        support_labels = episode_labels[support_indices]
        query_images = images[query_indices]
        query_labels = episode_labels[query_indices]

        # Extract features
        with torch.no_grad():
            support_features = model_base(support_images)
            query_features = model_base(query_images)

            # Normalize features if they're not already
            support_features = F.normalize(support_features, p=2, dim=1)
            query_features = F.normalize(query_features, p=2, dim=1)

        # Compute prototypes (nearest centroid approach)
        prototypes = []
        for cls_idx in range(n_way):
            cls_mask = (support_labels == cls_idx)
            if not torch.any(cls_mask):
                continue
            prototype = support_features[cls_mask].mean(dim=0)
            prototypes.append(prototype)

        if len(prototypes) < n_way:
            continue

        prototypes = torch.stack(prototypes)

        # Compute distances and make predictions
        # Calculate Euclidean distance between query features and prototypes
        dists = torch.cdist(query_features, prototypes)
        _, predictions = torch.min(dists, dim=1)

        # Compute accuracy
        accuracy = (predictions == query_labels).float().mean().item()
        episode_accuracies.append(accuracy)

        # Store predictions and true labels for metrics
        all_predictions.extend(predictions.cpu().numpy())
        all_true_labels.extend(query_labels.cpu().numpy())

    # Compute overall metrics
    metrics = {}
    if episode_accuracies:
        metrics['accuracy'] = np.mean(episode_accuracies)

        # Compute additional metrics if we have enough data
        if len(all_predictions) > 10:
            metrics['precision'] = precision_score(all_true_labels, all_predictions, average='macro', zero_division=0)
            metrics['recall'] = recall_score(all_true_labels, all_predictions, average='macro', zero_division=0)
            metrics['f1'] = f1_score(all_true_labels, all_predictions, average='macro', zero_division=0)
            metrics['cohen_kappa'] = cohen_kappa_score(all_true_labels, all_predictions)
            metrics['mcc'] = matthews_corrcoef(all_true_labels, all_predictions)

        print(f"Overall accuracy: {metrics['accuracy']:.4f}")
    else:
        metrics['accuracy'] = 0.0
        print("Warning: No valid episodes for evaluation")

    return metrics


def compare_standard_models_for_few_shot(test_dataset, output_dir='./model_comparisons',
                                        n_way=5, k_shot_values=[1, 5], n_query=15, n_episodes=50,
                                        device=torch.device("cuda")):
    """Compare standard CNN/Transformer models in a few-shot learning setting"""

    os.makedirs(output_dir, exist_ok=True)

    # Define CNN models to compare
    cnn_models = [
        (models.resnet50(pretrained=True), "ResNet50"),
        (models.efficientnet_b0(pretrained=True), "EfficientNet-B0"),
        (models.mobilenet_v2(pretrained=True), "MobileNetV2"),
        (models.densenet121(pretrained=True), "DenseNet121")
    ]

    # Define Transformer models to compare
    transformer_models = [
        (models.vit_b_16(pretrained=True), "ViT-B-16"),
        (timm.create_model("deit_small_patch16_224", pretrained=True), "DeiT-Small"),
    ]

    # Try to add more models if available
    try:
        cnn_models.append((models.vgg16(pretrained=True), "VGG16"))
    except:
        print("VGG16 not available, skipping")
        
    try:
        transformer_models.append((models.swin_t(pretrained=True), "Swin-Tiny"))
    except:
        print("Swin-Tiny not available, skipping")

    # Combine all models
    all_models = cnn_models + transformer_models

    # Results storage
    results = {}
    shot_results = {k: {} for k in k_shot_values}

    # Process each model
    for model_base, model_name in all_models:
        print(f"\nEvaluating {model_name} for few-shot learning...")

        # Remove the classification head
        if "vit" in model_name.lower() or "swin" in model_name.lower():
            if hasattr(model_base, 'head'):
                feature_dim = model_base.head.in_features
                model_base.head = nn.Identity()
            else:
                print(f"Warning: Could not identify feature dimension for {model_name}")
                continue
        elif "deit" in model_name.lower():
            if hasattr(model_base, 'head'):
                feature_dim = model_base.head.in_features
                model_base.head = nn.Identity()
            else:
                print(f"Warning: Could not identify feature dimension for {model_name}")
                continue
        elif hasattr(model_base, 'fc'):
            feature_dim = model_base.fc.in_features
            model_base.fc = nn.Identity()
        elif hasattr(model_base, 'classifier'):
            if isinstance(model_base.classifier, nn.Sequential):
                # Find the first Linear layer in the sequence
                for module in model_base.classifier:
                    if isinstance(module, nn.Linear):
                        feature_dim = module.in_features
                        break
                else:
                    # If no Linear layer found, try to infer from the model name
                    if 'vgg' in model_name.lower():
                        feature_dim = 25088  # VGG default
                    elif 'densenet' in model_name.lower():
                        feature_dim = 1024  # DenseNet default
                    else:
                        print(f"Warning: Could not identify feature dimension for {model_name}")
                        continue
            else:
                feature_dim = model_base.classifier.in_features
            model_base.classifier = nn.Identity()
        else:
            print(f"Warning: Could not identify feature dimension for {model_name}")
            continue

        # Move model to device
        model_base = model_base.to(device)
        model_base.eval()  # Use in evaluation mode as feature extractor

        # Evaluate for each shot value
        for k_shot in k_shot_values:
            print(f"  Evaluating {n_way}-way {k_shot}-shot...")
            # Evaluate in few-shot setting
            metrics = evaluate_standard_model_few_shot(
                model_base, test_dataset, n_way, k_shot, n_query, n_episodes, device
            )
            shot_results[k_shot][model_name] = metrics

        # Store average accuracy across shot values
        results[model_name] = np.mean([shot_results[k][model_name]['accuracy'] for k in k_shot_values])

    # Save results to CSV
    results_df = pd.DataFrame(columns=['Model', 'Type'] + [f'{k}-shot Accuracy' for k in k_shot_values] + ['Average Accuracy'])

    row_idx = 0
    for model_type, model_list in [("CNN", cnn_models), ("Transformer", transformer_models)]:
        for _, model_name in model_list:
            row = [model_name, model_type]
            for k in k_shot_values:
                row.append(shot_results[k][model_name]['accuracy'])
            row.append(results[model_name])
            results_df.loc[row_idx] = row
            row_idx += 1

    results_df.to_csv(os.path.join(output_dir, 'model_comparison_results.csv'), index=False)

    # Create comparison plots
    create_comparison_plots(results, shot_results, output_dir, k_shot_values)

    return results, shot_results


def evaluate_sprout_model(model_path, test_dataset, n_way=5, k_shot_values=[1, 5], n_query=15,
                      n_episodes=50, device=torch.device("cuda")):
    """
    Evaluate a trained SPROUT model for few-shot learning
    """
    # Load model
    num_classes = len(test_dataset.get_classes())
    model = SPROUT(
        num_classes=num_classes,
        backbone='resnet50',
        embed_dim=128,
        hidden_dims=[512, 256],
        num_refinement_steps=3
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    results = {}

    for k_shot in k_shot_values:
        print(f"Evaluating SPROUT model with {k_shot}-shot...")

        # Create episode sampler
        sampler = EpisodeBatchSampler(
            test_dataset, n_way=n_way, k_shot=k_shot+n_query, n_query=0, n_episodes=n_episodes
        )

        # Create dataloader
        dataloader = DataLoader(
            test_dataset, batch_sampler=sampler, num_workers=4, collate_fn=episode_collate
        )

        episode_accuracies = []
        all_predictions = []
        all_true_labels = []

        for batch in tqdm(dataloader, desc=f"SPROUT {k_shot}-shot evaluation"):
            images, labels = batch
            images = images.to(device)

            # Process similar to standard model evaluation
            unique_classes = torch.unique(labels)
            if len(unique_classes) < n_way:
                continue

            label_mapping = {unique_classes[i].item(): i for i in range(n_way)}
            episode_labels = torch.tensor([label_mapping[l.item()] for l in labels if l.item() in label_mapping],
                                         device=device)

            support_indices = []
            query_indices = []

            for cls_idx in range(n_way):
                cls_indices = torch.nonzero(episode_labels == cls_idx, as_tuple=True)[0]
                if len(cls_indices) < k_shot + 1:
                    continue

                support_indices.extend(cls_indices[:k_shot].tolist())
                query_indices.extend(cls_indices[k_shot:k_shot+min(n_query, len(cls_indices)-k_shot)].tolist())

            if len(support_indices) < n_way*k_shot or len(query_indices) == 0:
                continue

            support_images = images[support_indices]
            support_labels = episode_labels[support_indices]
            query_images = images[query_indices]
            query_labels = episode_labels[query_indices]

            # Evaluate
            with torch.no_grad():
                # Forward pass
                logits, _ = model(query_images, support_images, support_labels)

                # Calculate accuracy
                _, predicted = torch.max(logits, 1)
                accuracy = (predicted == query_labels).float().mean().item()
                episode_accuracies.append(accuracy)

                # Store predictions and true labels for metrics
                all_predictions.extend(predicted.cpu().numpy())
                all_true_labels.extend(query_labels.cpu().numpy())

        # Compute metrics
        metrics = {}
        if episode_accuracies:
            metrics['accuracy'] = np.mean(episode_accuracies)

            # Additional metrics
            if len(all_predictions) > 10:
                metrics['precision'] = precision_score(all_true_labels, all_predictions, average='macro', zero_division=0)
                metrics['recall'] = recall_score(all_true_labels, all_predictions, average='macro', zero_division=0)
                metrics['f1'] = f1_score(all_true_labels, all_predictions, average='macro', zero_division=0)
                metrics['cohen_kappa'] = cohen_kappa_score(all_true_labels, all_predictions)
                metrics['mcc'] = matthews_corrcoef(all_true_labels, all_predictions)

            print(f"SPROUT {k_shot}-shot accuracy: {metrics['accuracy']:.4f}")
        else:
            metrics['accuracy'] = 0.0
            print(f"Warning: No valid episodes for {k_shot}-shot evaluation")

        results[k_shot] = metrics

    return results


def create_comparison_plots(results, shot_results, output_dir, k_shot_values):
    """Create plots comparing different models"""

    # 1. Bar chart of average accuracy
    plt.figure(figsize=(14, 8))

    # Sort models by accuracy
    sorted_models = sorted(results.items(), key=lambda x: x[1], reverse=True)

    names = [model[0] for model in sorted_models]
    accs = [model[1] for model in sorted_models]

    # Color by model type
    colors = []
    for name in names:
        if any(t in name.lower() for t in ['vit', 'swin', 'deit']):
            colors.append('darkblue')  # Transformer
        else:
            colors.append('darkred')   # CNN

    bars = plt.bar(names, accs, color=colors)
    plt.ylabel('Average Accuracy Across Shot Values')
    plt.title('Few-Shot Learning Performance Comparison')
    plt.xticks(rotation=45, ha='right')

    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='darkred', label='CNN Models'),
        Patch(facecolor='darkblue', label='Transformer Models')
    ]
    plt.legend(handles=legend_elements)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'average_accuracy.pdf'))
    plt.close()

    # 2. Grouped bar chart for different shot values
    plt.figure(figsize=(14, 8))

    # Prepare data
    x = np.arange(len(names))
    width = 0.8 / len(k_shot_values)

    # Plot bars for each shot value
    for i, k in enumerate(k_shot_values):
        k_accs = [shot_results[k][name]['accuracy'] for name in names]
        offset = width * (i - len(k_shot_values)/2 + 0.5)
        bars = plt.bar(x + offset, k_accs, width, label=f'{k}-shot')

    plt.ylabel('Accuracy')
    plt.title('Few-Shot Learning Performance by Shot Value')
    plt.xticks(x, names, rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shot_comparison.pdf'))
    plt.close()

    # 3. Separate CNN vs Transformer performance
    plt.figure(figsize=(10, 6))

    # Split by model type
    cnn_names = [name for name in names if not any(t in name.lower() for t in ['vit', 'swin', 'deit'])]
    transformer_names = [name for name in names if any(t in name.lower() for t in ['vit', 'swin', 'deit'])]

    cnn_accs = [results[name] for name in cnn_names]
    transformer_accs = [results[name] for name in transformer_names]

    # Box plot
    plt.boxplot([cnn_accs, transformer_accs], labels=['CNN Models', 'Transformer Models'])
    plt.ylabel('Average Accuracy')
    plt.title('CNN vs Transformer Performance')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cnn_vs_transformer.pdf'))
    plt.close()


def compare_sprout_vs_standard_models(test_dataset, sprout_model_path, output_dir='./comparison_results',
                                  n_way=5, k_shot_values=[1, 5], n_query=15, n_episodes=50,
                                  device=torch.device("cuda")):
    """
    Compare SPROUT model against standard CNN and Transformer models
    """
    os.makedirs(output_dir, exist_ok=True)

    # Evaluate standard models
    print(f"\nEvaluating standard models...")
    _, standard_results = compare_standard_models_for_few_shot(
        test_dataset, output_dir, n_way, k_shot_values, n_query, n_episodes, device
    )

    # Evaluate SPROUT model
    print(f"\nEvaluating SPROUT model...")
    sprout_results = evaluate_sprout_model(
        sprout_model_path, test_dataset, n_way, k_shot_values, n_query, n_episodes, device
    )

    # Create SPROUT comparison plots
    create_sprout_comparison_plots(standard_results, sprout_results, output_dir, k_shot_values, n_way)

    # Create comprehensive result table
    create_comprehensive_results_table(standard_results, sprout_results, output_dir, k_shot_values)

    return standard_results, sprout_results


def create_sprout_comparison_plots(standard_results, sprout_results, output_dir, k_shot_values, n_way):
    """
    Create visualizations comparing SPROUT with standard models
    """
    for k_shot in k_shot_values:
        # Collect accuracies
        model_names = list(standard_results[k_shot].keys()) + ['SPROUT (Ours)']
        accuracies = [standard_results[k_shot][name]['accuracy'] for name in model_names[:-1]]
        accuracies.append(sprout_results[k_shot]['accuracy'])

        # Sort by accuracy
        sorted_indices = np.argsort(accuracies)[::-1]
        sorted_names = [model_names[i] for i in sorted_indices]
        sorted_accs = [accuracies[i] for i in sorted_indices]

        # Color by model type
        colors = []
        for name in sorted_names:
            if name == 'SPROUT (Ours)':
                colors.append('darkgreen')  # SPROUT
            elif any(t in name.lower() for t in ['vit', 'swin', 'deit']):
                colors.append('darkblue')  # Transformer
            else:
                colors.append('darkred')   # CNN

        # Create bar chart
        plt.figure(figsize=(14, 8))
        bars = plt.bar(sorted_names, sorted_accs, color=colors)
        plt.ylabel(f'{k_shot}-shot Accuracy')
        plt.title(f'{n_way}-way {k_shot}-shot Performance Comparison')
        plt.xticks(rotation=45, ha='right')

        # Add value labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='darkred', label='CNN Models'),
            Patch(facecolor='darkblue', label='Transformer Models'),
            Patch(facecolor='darkgreen', label='SPROUT (Ours)')
        ]
        plt.legend(handles=legend_elements)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{k_shot}_shot_comparison.pdf'))
        plt.close()

    # Create performance improvement chart
    plt.figure(figsize=(10, 8))

    improvements = []
    for k_shot in k_shot_values:
        # Find best standard model accuracy
        best_standard_acc = max([standard_results[k_shot][name]['accuracy'] for name in standard_results[k_shot].keys()])
        sprout_acc = sprout_results[k_shot]['accuracy']

        # Calculate improvement percentage
        improvement = (sprout_acc - best_standard_acc) / best_standard_acc * 100
        improvements.append(improvement)

    plt.bar(k_shot_values, improvements, color='darkgreen')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.ylabel('Improvement Over Best Standard Model (%)')
    plt.xlabel('Number of Shots (K)')
    plt.title('SPROUT Improvement Over Standard Models')

    # Add value labels
    for i, v in enumerate(improvements):
        plt.text(k_shot_values[i], v + (1 if v >= 0 else -3),
                f'{v:.1f}%', ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sprout_improvement.pdf'))
    plt.close()


def create_comprehensive_results_table(standard_results, sprout_results, output_dir, k_shot_values):
    """
    Create a comprehensive table of all results
    """
    # Initialize DataFrame
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'cohen_kappa', 'mcc']
    columns = ['Model', 'Type'] + [f'{k}-shot {m.capitalize()}' for k in k_shot_values for m in metrics]

    results_df = pd.DataFrame(columns=columns)

    # Add standard models
    row_idx = 0
    for model_name in standard_results[k_shot_values[0]].keys():
        # Determine model type
        model_type = "Transformer" if any(t in model_name.lower() for t in ['vit', 'swin', 'deit']) else "CNN"

        row = [model_name, model_type]

        # Add metrics for each shot value
        for k in k_shot_values:
            for metric in metrics:
                if metric in standard_results[k][model_name]:
                    row.append(standard_results[k][model_name][metric])
                else:
                    row.append(np.nan)

        results_df.loc[row_idx] = row
        row_idx += 1

    # Add SPROUT model
    row = ['SPROUT (Ours)', 'Few-Shot Learning']
    for k in k_shot_values:
        for metric in metrics:
            if metric in sprout_results[k]:
                row.append(sprout_results[k][metric])
            else:
                row.append(np.nan)

    results_df.loc[row_idx] = row

    # Save to CSV
    results_df.to_csv(os.path.join(output_dir, 'comprehensive_results.csv'), index=False)

    return results_df


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Compare SPROUT against standard models for few-shot learning')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained SPROUT model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='./comparison_results', help='Output directory for results')
    parser.add_argument('--n_way', type=int, default=5, help='Number of classes per episode')
    parser.add_argument('--k_shot', type=int, nargs='+', default=[1, 5, 10], help='Number of support examples per class')
    parser.add_argument('--n_query', type=int, default=15, help='Number of query examples per class')
    parser.add_argument('--n_episodes', type=int, default=50, help='Number of episodes to evaluate')
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

    # Run comparison
    standard_results, sprout_results = compare_sprout_vs_standard_models(
        test_dataset, args.model_path, args.output_dir,
        args.n_way, args.k_shot, args.n_query, args.n_episodes, device
    )

    print(f"Comparison completed. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()