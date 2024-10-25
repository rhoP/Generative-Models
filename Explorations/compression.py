import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import cv2
from PIL import Image
import io


# Function to compute Total Variation Distance between two images
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


# Function to compute SSIM between two images
def compute_ssim(img1, img2):
    img1_np = img1.cpu().numpy().transpose(1, 2, 0)  # Convert to numpy array
    img2_np = img2.cpu().numpy().transpose(1, 2, 0)
    img1_np = (img1_np * 255).astype(np.uint8)
    img2_np = (img2_np * 255).astype(np.uint8)
    ssim_value, _ = ssim(img1_np, img2_np, full=True, channel_axis=2)
    return ssim_value


# Function to apply JPEG compression and return the compressed image
def compress_image(image_tensor, quality):
    # Convert to PIL image
    img_pil = TF.to_pil_image(image_tensor.cpu())

    # Save image to buffer in JPEG format with specified quality
    buffer = io.BytesIO()
    img_pil.save(buffer, format="JPEG", quality=quality)

    # Load the compressed image from the buffer
    compressed_img_pil = Image.open(buffer)

    # Convert back to tensor
    compressed_img_tensor = TF.to_tensor(compressed_img_pil)
    return compressed_img_tensor


# Load a sample image (e.g., from CIFAR-10 or custom)
from torchvision import datasets, transforms

cifar10 = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
image, _ = cifar10[0]  # Get a sample image from CIFAR-10 dataset

# List of compression quality levels
compression_levels = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# Initialize lists to store SSIM and TV distances
ssim_scores = []
tv_distances = []

# Compute SSIM and TV for different compression levels
for quality in compression_levels:
    # Compress image
    compressed_image = compress_image(image, quality)

    # Compute SSIM
    ssim_value = compute_ssim(image, compressed_image)
    ssim_scores.append(ssim_value)

    # Compute Total Variation Distance
    tv_distance = total_variation_norm(image.unsqueeze(0), compressed_image.unsqueeze(0))
    tv_distances.append(tv_distance)

# Plot the results
plt.figure(figsize=(12, 6))

# SSIM vs Compression Level
plt.subplot(1, 2, 1)
plt.plot(compression_levels, ssim_scores, marker='o', color='blue')
plt.title("SSIM vs Compression Level")
plt.xlabel("JPEG Compression Quality")
plt.ylabel("SSIM")
plt.grid(True)

# Total Variation Distance vs Compression Level
plt.subplot(1, 2, 2)
plt.plot(compression_levels, tv_distances, marker='x', color='red')
plt.title("Total Variation Distance vs Compression Level")
plt.xlabel("JPEG Compression Quality")
plt.ylabel("Total Variation Distance")
plt.grid(True)

plt.tight_layout()
plt.show()
