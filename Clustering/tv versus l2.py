import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.cluster import KMeans


# Function to compute the Total Variation Norm for a batch of 2D images
def batch_total_variation_norm(images):
    padded_images = F.pad(images, (1, 1, 1, 1), mode='replicate')

    # Calculate differences with neighbors
    diff_top = torch.abs(padded_images[:, :, 1:-1, 1:-1] - padded_images[:, :, :-2, 1:-1])
    diff_bottom = torch.abs(padded_images[:, :, 1:-1, 1:-1] - padded_images[:, :, 2:, 1:-1])
    diff_left = torch.abs(padded_images[:, :, 1:-1, 1:-1] - padded_images[:, :, 1:-1, :-2])
    diff_right = torch.abs(padded_images[:, :, 1:-1, 1:-1] - padded_images[:, :, 1:-1, 2:])

    # Average of absolute differences
    pointwise_variation = (diff_top + diff_bottom + diff_left + diff_right) / 4.0

    # Return the maximum variation for each image
    tv_norm = torch.max(pointwise_variation.view(images.size(0), -1), dim=1)[0]
    return tv_norm


# Function to compute the Frobenius norm for a batch of images
def batch_frobenius_norm(images):
    # Flatten the images and compute the Frobenius norm (L2 norm of all pixels)
    return torch.norm(images.view(images.size(0), -1), p=2, dim=1)


# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=1024, shuffle=False)

# Move dataset to GPU and compute TV norm and Frobenius norm
features = []
labels = []

for images, targets in train_loader:
    images = images.to(device)  # Move images to GPU

    # Compute TV norm for the batch
    tv_norm_batch = batch_total_variation_norm(images)

    # Compute Frobenius norm for the batch
    frobenius_norm_batch = batch_frobenius_norm(images)

    # Store the features for both norms
    features.append(torch.cat((tv_norm_batch.view(-1, 1), frobenius_norm_batch.view(-1, 1)),
                              dim=1).cpu().numpy())
    labels.append(targets.cpu().numpy())

# Convert the results to numpy arrays for further processing
features = np.concatenate(features)
labels = np.concatenate(labels)

# Clustering using KMeans
kmeans = KMeans(n_clusters=10, random_state=42)
cluster_labels = kmeans.fit_predict(features)

# Dimensionality Reduction for Visualization (t-SNE)
print("clustering done")
tsne = TSNE(n_components=2, random_state=42)
tsne_result = tsne.fit_transform(features)

# Plot the clusters
plt.figure(figsize=(10, 8))
scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=cluster_labels, cmap='tab10', s=5)
plt.colorbar(scatter, ticks=range(10))
plt.title("MNIST Clusters")
plt.xlabel("tv")
plt.ylabel("frob")
plt.show()
