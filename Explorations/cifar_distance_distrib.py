import torch
import torchvision
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from torchvision import transforms
import cv2

# Function to compute the Total Variation distance between two images
def total_variation_norm(img1, img2):
    diff_img = torch.abs(img1 - img2)
    padded_img = F.pad(diff_img, (1, 1, 1, 1), mode='replicate')
    diff_top = torch.abs(padded_img[:, :, 1:-1, 1:-1] - padded_img[:, :, :-2, 1:-1])
    diff_bottom = torch.abs(padded_img[:, :, 1:-1, 1:-1] - padded_img[:, :, 2:, 1:-1])
    diff_left = torch.abs(padded_img[:, :, 1:-1, 1:-1] - padded_img[:, :, 1:-1, :-2])
    diff_right = torch.abs(padded_img[:, :, 1:-1, 1:-1] - padded_img[:, :, 1:-1, 2:])
    pointwise_variation = (diff_top + diff_bottom + diff_left + diff_right) / 4.0
    tv_norm = torch.max(pointwise_variation.view(diff_img.size(0), -1), dim=1)[0]
    return tv_norm.item()

# Function to compute the SSIM between two images
def compute_ssim(img1, img2):
    img1_np = img1.cpu().numpy().transpose(1, 2, 0)  # Convert to numpy array
    img2_np = img2.cpu().numpy().transpose(1, 2, 0)
    img1_np = (img1_np * 255).astype(np.uint8)
    img2_np = (img2_np * 255).astype(np.uint8)
    # print(img1_np.shape)
    ssim_value, _ = ssim(img1_np, img2_np, full=True, channel_axis=2)
    return ssim_value

# Load CIFAR-10 dataset and apply basic transformation
transform = transforms.Compose([
    transforms.ToTensor()  # Convert images to PyTorch tensors
])

# Load the CIFAR-10 dataset (train subset)
cifar10 = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Use a small subset to reduce memory usage (e.g., 10 images per class)
class_indices = {i: [] for i in range(20)}  # Store indices by class
for idx, (_, label) in enumerate(cifar10):
    if len(class_indices[label]) < 10:  # Limit to 10 images per class
        class_indices[label].append(idx)
    if all(len(v) == 10 for v in class_indices.values()):  # Stop once we have enough
        break

# Flatten the indices into one list
subset_indices = [idx for indices in class_indices.values() for idx in indices]
cifar10_subset = Subset(cifar10, subset_indices)
dataloader = DataLoader(cifar10_subset, batch_size=1, shuffle=False, num_workers=2)

# Initialize lists to store distances
tv_same_class = []
tv_diff_class = []
ssim_same_class = []
ssim_diff_class = []

# Pairwise comparison of images
images = []
labels = []

# Load all images and labels
for img, label in dataloader:
    images.append(img.squeeze())  # Remove the batch dimension
    labels.append(label.item())

# Compare each image pair
for i in range(len(images)):
    for j in range(i+1, len(images)):  # Avoid duplicate comparisons
        img1 = images[i]
        img2 = images[j]
        label1 = labels[i]
        label2 = labels[j]

        # Compute Total Variation distance and SSIM
        tv_distance = total_variation_norm(img1.unsqueeze(0), img2.unsqueeze(0))
        ssim_value = compute_ssim(img1, img2)

        # Store the distances in respective lists
        if label1 == label2:
            tv_same_class.append(tv_distance)
            ssim_same_class.append(ssim_value)
        else:
            tv_diff_class.append(tv_distance)
            ssim_diff_class.append(ssim_value)

# Plot the distributions
plt.figure(figsize=(12, 6))

# Total Variation norm distribution
plt.subplot(1, 2, 1)
plt.hist(tv_same_class, bins=20, alpha=0.7, label="Same Class", color='red')
plt.hist(tv_diff_class, bins=20, alpha=0.7, label="Different Class", color='green')
plt.title("Total Variation Distance Distribution")
plt.xlabel("Total Variation Distance")
plt.ylabel("Frequency")
plt.legend()

# SSIM distribution
plt.subplot(1, 2, 2)
plt.hist(ssim_same_class, bins=20, alpha=0.7, label="Same Class", color='red')
plt.hist(ssim_diff_class, bins=20, alpha=0.7, label="Different Class", color='green')
plt.title("SSIM Distribution")
plt.xlabel("SSIM")
plt.ylabel("Frequency")
plt.legend()

plt.tight_layout()
plt.show()
