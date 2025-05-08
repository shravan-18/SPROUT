import random
import torch
from torch.utils.data import Sampler

class EpisodeBatchSampler(Sampler):
    """
    Sampler for creating few-shot episode batches
    
    This sampler creates episodes for few-shot learning, where each episode
    consists of N classes with K support examples and Q query examples per class.
    """
    def __init__(self, dataset, n_way, k_shot, n_query, n_episodes):
        """
        Args:
            dataset: Dataset to sample from
            n_way: Number of classes per episode
            k_shot: Number of support examples per class
            n_query: Number of query examples per class
            n_episodes: Number of episodes to create
        """
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.n_episodes = n_episodes
        self.indices_by_class = dataset.get_indices_by_class()
        self.classes = list(self.indices_by_class.keys())

        # Ensure we have enough classes
        if len(self.classes) < n_way:
            raise ValueError(f"Not enough classes. Requested {n_way}, but only {len(self.classes)} available.")

    def __iter__(self):
        for _ in range(self.n_episodes):
            # Sample N classes for this episode
            episode_classes = random.sample(self.classes, self.n_way)

            # Collect indices for the episode
            episode_indices = []

            for c in episode_classes:
                # Get all indices for this class
                class_indices = self.indices_by_class[c]

                # Make sure we have enough samples
                samples_needed = self.k_shot + self.n_query
                if len(class_indices) < samples_needed:
                    # If not enough, sample with replacement
                    selected_indices = random.choices(class_indices, k=samples_needed)
                else:
                    # Sample without replacement
                    selected_indices = random.sample(class_indices, samples_needed)

                # Add to episode indices
                episode_indices.extend(selected_indices)

            # Yield the episode batch indices
            yield episode_indices

    def __len__(self):
        return self.n_episodes


class EpisodeSampler:
    """
    Sampler for creating few-shot episodes with more flexibility
    
    This sampler provides more control over the episode creation process,
    allowing for separate access to support and query sets.
    """
    def __init__(self, dataset, n_way=5, k_shot=5, n_query=15, n_episodes=100):
        """
        Args:
            dataset: Dataset to sample from
            n_way: Number of classes per episode
            k_shot: Number of support examples per class
            n_query: Number of query examples per class
            n_episodes: Number of episodes to create
        """
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.n_episodes = n_episodes

        # Group indices by class
        self.label_indices = {}
        for idx, (_, label) in enumerate(dataset):
            if label.item() not in self.label_indices:
                self.label_indices[label.item()] = []
            self.label_indices[label.item()].append(idx)

    def sample_episode(self):
        """
        Sample a single episode
        
        Returns:
            support_indices: Indices of support set
            query_indices: Indices of query set
            episode_classes: Classes in this episode
        """
        # Sample N classes
        available_classes = list(self.label_indices.keys())
        if len(available_classes) < self.n_way:
            raise ValueError(f"Not enough classes. Requested {self.n_way}, but only {len(available_classes)} available.")

        episode_classes = random.sample(available_classes, self.n_way)

        support_indices = []
        query_indices = []

        for class_idx, c in enumerate(episode_classes):
            # Get all indices for this class
            class_indices = self.label_indices[c]

            # Check if we have enough examples
            if len(class_indices) < self.k_shot + self.n_query:
                # If not enough, sample with replacement
                support_idx = random.choices(class_indices, k=self.k_shot)
                remaining = random.choices(class_indices, k=self.n_query)
            else:
                # Sample without replacement
                samples = random.sample(class_indices, self.k_shot + self.n_query)
                support_idx = samples[:self.k_shot]
                remaining = samples[self.k_shot:]

            # Add to support and query sets
            support_indices.extend(support_idx)
            query_indices.extend(remaining)

        return support_indices, query_indices, episode_classes

    def __iter__(self):
        """
        Iterator for sampling episodes
        """
        for _ in range(self.n_episodes):
            yield self.sample_episode()


def episode_collate(batch):
    """
    Collate function for episode batches
    
    Args:
        batch: List of (image, label) tuples
        
    Returns:
        all_images: Tensor of all images in the batch
        all_labels: Tensor of all labels in the batch
    """
    all_images = []
    all_labels = []

    for img, label in batch:
        all_images.append(img)
        all_labels.append(label)

    all_images = torch.stack(all_images)
    all_labels = torch.tensor(all_labels)

    return all_images, all_labels


def create_episode_batch(dataset, support_indices, query_indices, episode_classes):
    """
    Create batches for an episode
    
    Args:
        dataset: Dataset to sample from
        support_indices: Indices of support set
        query_indices: Indices of query set
        episode_classes: Classes in this episode
        
    Returns:
        support_images: Support set images
        support_labels: Support set labels
        query_images: Query set images
        query_labels: Query set labels
    """
    # Map original class indices to episode class indices (0 to N-1)
    class_map = {c: i for i, c in enumerate(episode_classes)}

    # Create support set
    support_images = []
    support_labels = []

    for idx in support_indices:
        image, label = dataset[idx]
        support_images.append(image)
        # Map to episode class index
        support_labels.append(class_map[label.item()])

    # Create query set
    query_images = []
    query_labels = []

    for idx in query_indices:
        image, label = dataset[idx]
        query_images.append(image)
        # Map to episode class index
        query_labels.append(class_map[label.item()])

    # Convert to tensors
    support_images = torch.stack(support_images)
    support_labels = torch.tensor(support_labels)
    query_images = torch.stack(query_images)
    query_labels = torch.tensor(query_labels)

    return support_images, support_labels, query_images, query_labels