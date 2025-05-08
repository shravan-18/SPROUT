# Visualizations

This directory contains visualizations generated during training and evaluation of the SPROUT model.

## Types of Visualizations

### Embedding Space Visualization

t-SNE visualization of the embedding space, showing how different disease classes are separated in the learned metric space.

### Prototype Refinement Visualization

Shows how prototypes evolve during the refinement process, demonstrating how the attention mechanism focuses on disease-relevant features.

### Distance Heatmap

Visualizes distances between query samples and prototypes, providing insight into how the model makes classification decisions.

### Shot Analysis

Performance analysis across different shot values (1-shot, 5-shot, etc.), showing how the model adapts to increasing amounts of support examples.

### Model Comparison

Comparisons between SPROUT and standard CNN/Transformer models in few-shot learning scenarios.

## Generated Files

- `embedding_space_{n_way}way_{k_shot}shot.png`: t-SNE visualization of embeddings
- `confusion_matrix_{n_way}way_{k_shot}shot.png`: Confusion matrix for evaluation results
- `distance_heatmap_{n_way}way_{k_shot}shot.png`: Distance heatmap between query samples and prototypes
- `shot_comparison_{n_way}way.png`: Accuracy comparison across different shot values
- `model_comparison_results.pdf`: Comparison of SPROUT with standard models
- `prototype_refinement.png`: Visualization of prototype evolution during refinement