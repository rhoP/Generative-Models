import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import os
from torchvision import transforms


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


# Load BSDS500 or any other high-resolution image dataset
# Assuming the images are stored in './bsds500/' directory
image_dir = '../../data/BSDS300/images/train/'
image_filenames = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]
# Select a subset of 50 or more images
num_images = 50  # You can increase this to any number of images
selected_filenames = image_filenames[:num_images]

# Transform for converting images to tensors
transform = transforms.ToTensor()

# List of compression quality levels
compression_levels = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# Initialize a list to store TV distances for all images
tv_distances_all = []

# Loop through each image, compress, and compute TV distances
for image_file in image_filenames:
    image_path = os.path.join(image_dir, image_file)
    original_image = Image.open(image_path)
    transform = transforms.ToTensor()
    image_tensor = transform(original_image).unsqueeze(0)  # Convert image to tensor and add batch dimension

    tv_distances = []

    for quality in compression_levels:
        # Compress image
        compressed_image = compress_image(image_tensor.squeeze(0), quality)

        # Compute Total Variation Distance
        tv_distance = total_variation_norm(image_tensor, compressed_image.unsqueeze(0))
        tv_distances.append(tv_distance)

    # Store the TV distances for this image
    tv_distances_all.append(tv_distances)

# Plot the results
plt.figure(figsize=(10, 6))

# Plot TV distance curves for all images
for tv_distances in tv_distances_all:
    plt.scatter(compression_levels, tv_distances, marker='x', alpha=0.7)

plt.title("Total Variation Distance vs Compression Quality for Multiple Images")
plt.xlabel("JPEG Compression Quality")
plt.ylabel("Total Variation Distance")
plt.grid(True)

plt.tight_layout()
plt.show()