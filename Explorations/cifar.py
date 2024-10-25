import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans


# Function to compute the Total Variation Norm for each channel in a batch of images
def batch_total_variation_norm_rgb(images):
    # Split the image into R, G, B channels
    r_channel = images[:, 0:1, :, :]
    g_channel = images[:, 1:2, :, :]
    b_channel = images[:, 2:2 + 1, :, :]

    # Compute TV norm for each channel separately
    tv_r = batch_total_variation_norm_single_channel(r_channel)
    tv_g = batch_total_variation_norm_single_channel(g_channel)
    tv_b = batch_total_variation_norm_single_channel(b_channel)

    return tv_r, tv_g, tv_b


# Function to compute TV norm for a single channel
def batch_total_variation_norm_single_channel(channel):
    padded_channel = F.pad(channel, (1, 1, 1, 1), mode='replicate')

    # Calculate differences with neighbors
    diff_top = torch.abs(padded_channel[:, :, 1:-1, 1:-1] - padded_channel[:, :, :-2, 1:-1])
    diff_bottom = torch.abs(padded_channel[:, :, 1:-1, 1:-1] - padded_channel[:, :, 2:, 1:-1])
    diff_left = torch.abs(padded_channel[:, :, 1:-1, 1:-1] - padded_channel[:, :, 1:-1, :-2])
    diff_right = torch.abs(padded_channel[:, :, 1:-1, 1:-1] - padded_channel[:, :, 1:-1, 2:])

    # Average of absolute differences
    pointwise_variation = (diff_top + diff_bottom + diff_left + diff_right) / 4.0

    # Return the maximum variation for each image
    tv_norm = torch.max(pointwise_variation.view(channel.size(0), -1), dim=1)[0]
    return tv_norm


# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load CIFAR-10 dataset
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=1024, shuffle=False)

# Store features for all images
tv_r_features = []
tv_g_features = []
tv_b_features = []
labels = []

# Iterate over the dataset and compute TV norms for each channel
for images, targets in train_loader:
    images = images.to(device)  # Move images to GPU

    # Compute TV norm for R, G, B channels
    tv_r_batch, tv_g_batch, tv_b_batch = batch_total_variation_norm_rgb(images)

    # Store the features for all three channels
    tv_r_features.append(tv_r_batch.cpu().numpy())
    tv_g_features.append(tv_g_batch.cpu().numpy())
    tv_b_features.append(tv_b_batch.cpu().numpy())

    labels.append(targets.cpu().numpy())

# Convert the results to numpy arrays for further processing
tv_r_features = np.concatenate(tv_r_features)
tv_g_features = np.concatenate(tv_g_features)
tv_b_features = np.concatenate(tv_b_features)
labels = np.concatenate(labels)

# Stack TV norms for R, G, B channels into a 2D array (for t-SNE)
tv_rgb_features = np.stack([tv_r_features, tv_g_features, tv_b_features], axis=1)

# Dimensionality Reduction for Visualization (t-SNE)
tsne = TSNE(n_components=2, random_state=42)
tsne_result = tsne.fit_transform(tv_rgb_features)

# Explorations using KMeans (optional, for color-coding based on clusters)
kmeans = KMeans(n_clusters=10, random_state=42)
cluster_labels = kmeans.fit_predict(tv_rgb_features)

# Plot the t-SNE visualization
plt.figure(figsize=(8, 6))
scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=cluster_labels, cmap='tab10', s=5)
plt.title("t-SNE of CIFAR-10 using TV Norm on RGB Channels")
plt.colorbar(scatter, ticks=range(10))
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.show()
