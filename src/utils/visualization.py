import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

def plot_embeddings_tsne(embeddings, labels, prototypes=None, title="t-SNE Visualization of Embeddings"):
    """
    Plot t-SNE visualization of embeddings
    
    Args:
        embeddings: Embeddings to visualize
        labels: Labels for coloring
        prototypes: Class prototypes (optional)
        title: Plot title
    """
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Create plot
    plt.figure(figsize=(10, 8))

    # Plot embeddings
    unique_labels = np.unique(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                    c=[colors[i]], label=f"Class {label}", alpha=0.7)

    # Plot prototypes if provided
    if prototypes is not None:
        prototypes_2d = tsne.fit_transform(np.vstack([embeddings, prototypes]))[-len(prototypes):]
        plt.scatter(prototypes_2d[:, 0], prototypes_2d[:, 1],
                    c='black', marker='*', s=200, label='Prototypes')

    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


def plot_confusion_matrix(true_labels, pred_labels, title="Confusion Matrix"):
    """
    Plot confusion matrix
    
    Args:
        true_labels: Ground truth labels
        pred_labels: Predicted labels
        title: Plot title
    """
    # Compute confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)

    # Create plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.show()


def plot_prototype_evolution(initial_prototypes, refined_prototypes):
    """
    Visualize how prototypes evolve during refinement
    
    Args:
        initial_prototypes: Initial prototypes
        refined_prototypes: Refined prototypes
    """
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    all_prototypes = np.vstack([initial_prototypes, refined_prototypes])
    all_prototypes_2d = tsne.fit_transform(all_prototypes)

    # Split back
    n_prototypes = len(initial_prototypes)
    initial_prototypes_2d = all_prototypes_2d[:n_prototypes]
    refined_prototypes_2d = all_prototypes_2d[n_prototypes:]

    # Create plot
    plt.figure(figsize=(10, 8))

    # Plot prototypes
    for i in range(n_prototypes):
        plt.scatter(initial_prototypes_2d[i, 0], initial_prototypes_2d[i, 1],
                    c='blue', marker='o', s=100, alpha=0.7)
        plt.scatter(refined_prototypes_2d[i, 0], refined_prototypes_2d[i, 1],
                    c='red', marker='*', s=200, alpha=0.7)

        # Draw arrow from initial to refined
        plt.arrow(initial_prototypes_2d[i, 0], initial_prototypes_2d[i, 1],
                 refined_prototypes_2d[i, 0] - initial_prototypes_2d[i, 0],
                 refined_prototypes_2d[i, 1] - initial_prototypes_2d[i, 1],
                 color='black', width=0.01, head_width=0.1)

    plt.scatter([], [], c='blue', marker='o', s=100, label='Initial Prototypes')
    plt.scatter([], [], c='red', marker='*', s=200, label='Refined Prototypes')

    plt.title("Prototype Evolution during Refinement")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


def plot_training_curves(train_accuracies, train_losses):
    """
    Plot training curves
    
    Args:
        train_accuracies: List of training accuracies
        train_losses: List of training losses
    """
    plt.figure(figsize=(12, 5))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(train_accuracies, 'b-', label='Training Accuracy')
    plt.title('Training Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(alpha=0.3)
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, 'r-', label='Training Loss')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_shot_comparison(shot_results):
    """
    Plot comparison of accuracies across different shot values
    
    Args:
        shot_results: Dictionary mapping shot values to accuracies
    """
    plt.figure(figsize=(10, 6))

    shots = list(shot_results.keys())
    accuracies = list(shot_results.values())

    plt.plot(shots, accuracies, 'bo-', linewidth=2, markersize=10)

    for shot, acc in zip(shots, accuracies):
        plt.text(shot, acc + 0.01, f'{acc:.3f}', ha='center')

    plt.title('Accuracy vs. Number of Shots')
    plt.xlabel('Number of Shots (K)')
    plt.ylabel('Accuracy')
    plt.grid(alpha=0.3)
    plt.xticks(shots)
    plt.ylim(top=1.05)
    plt.tight_layout()
    plt.show()


def visualize_distance_heatmap(query_embeddings, prototypes, query_labels):
    """
    Visualize distance heatmap between query samples and prototypes
    
    Args:
        query_embeddings: Query embeddings
        prototypes: Class prototypes
        query_labels: Query labels
    """
    # Compute distances
    distances = np.zeros((len(query_embeddings), len(prototypes)))

    for i, query in enumerate(query_embeddings):
        for j, proto in enumerate(prototypes):
            distances[i, j] = np.sum((query - proto) ** 2)

    # Sort by true label for better visualization
    sort_idx = np.argsort(query_labels)
    sorted_distances = distances[sort_idx]
    sorted_labels = query_labels[sort_idx]

    # Create plot
    plt.figure(figsize=(10, 8))

    # Plot heatmap
    sns.heatmap(sorted_distances, cmap='viridis_r')

    plt.title("Distance Heatmap: Query Samples vs. Prototypes")
    plt.xlabel("Prototype Index")
    plt.ylabel("Query Sample Index")

    # Add horizontal lines to separate classes
    prev_label = sorted_labels[0]
    for i, label in enumerate(sorted_labels[1:], 1):
        if label != prev_label:
            plt.axhline(y=i, color='red', linestyle='-', linewidth=1)
            prev_label = label

    plt.tight_layout()
    plt.show()