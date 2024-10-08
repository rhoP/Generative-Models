import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE


# Function to compute the Total Variation Norm for a batch of 2D images
def batch_total_variation_norm(images):
    """
    Compute the total variation (TV) norm for a batch of 2D images as defined by the user.

    The TV norm is the maximum of the point-wise variation, where the point-wise variation
    is the average of the absolute differences between each pixel's value and its neighbors.

    Parameters:
    images (torch.Tensor): Input 4D tensor (batch_size, 1, H, W).

    Returns:
    torch.Tensor: The total variation norm for each image in the batch.
    """
    # Pad the images to handle boundary conditions
    padded_images = F.pad(images, (1, 1, 1, 1), mode='replicate')

    # Calculate differences with neighbors (top, bottom, left, right)
    diff_top = torch.abs(padded_images[:, :, 1:-1, 1:-1] - padded_images[:, :, :-2, 1:-1])
    diff_bottom = torch.abs(padded_images[:, :, 1:-1, 1:-1] - padded_images[:, :, 2:, 1:-1])
    diff_left = torch.abs(padded_images[:, :, 1:-1, 1:-1] - padded_images[:, :, 1:-1, :-2])
    diff_right = torch.abs(padded_images[:, :, 1:-1, 1:-1] - padded_images[:, :, 1:-1, 2:])

    # Compute point-wise variation as the average of the absolute differences
    pointwise_variation = (diff_top + diff_bottom + diff_left + diff_right) / 4.0

    # Return the maximum variation for each image (the TV norm)
    tv_norm = torch.max(pointwise_variation.view(images.size(0), -1), dim=1)[0]
    return tv_norm


# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Load MNIST dataset (use transform to Tensor)
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=1024, shuffle=False)

# Move dataset to GPU and compute TV norm
features = []
labels = []

for images, targets in train_loader:
    images = images.to(device)  # Move images to GPU
    # Compute TV norm for the batch
    tv_norm_batch = batch_total_variation_norm(images)

    # More features
    # flat_images = images.view(images.size(0), -1)

    # combined_features = torch.cat((tv_norm_batch.view(-1, 1), flat_images), dim=1)

    # features.append(combined_features.cpu().numpy())  # Move back to CPU for further processing
    features.append(tv_norm_batch.view(-1).cpu().numpy())
    labels.append(targets.view(-1).cpu().numpy())

# Convert the results to numpy arrays for clustering
# print("converting results to numpy array")
# Convert the results to numpy arrays for clustering
features = np.concatenate(features)
labels = np.concatenate(labels)

# Clustering using KMeans
# kmeans = KMeans(n_clusters=10, random_state=42)
# cluster_labels = kmeans.fit_predict(features)

# Dimensionality Reduction for Visualization (t-SNE)
# print("clustering done")
# tsne = TSNE(n_components=2, random_state=42)
# tsne_result = tsne.fit_transform(features)
print("shapes: ", len(features), len(labels), "\t", features[0])
# Plot the clusters
plt.figure(figsize=(10, 8))
# scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=cluster_labels, cmap='tab10', s=5)
scatter = plt.scatter(features, labels, c=labels, cmap='tab10', s=5)
plt.plot
plt.colorbar(scatter, ticks=range(10))
plt.title("MNIST Clusters")
plt.xlabel("norms")
plt.ylabel("labels")
plt.show()
