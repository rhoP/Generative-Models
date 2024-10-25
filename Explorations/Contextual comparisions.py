import torch
import torchvision
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader

# Function to compute the L2 norm between two images
def l2_norm(img1, img2):
    return torch.norm(img1 - img2, p=2).item()

# Function to compute the Total Variation norm (TV norm) between two images
def total_variation_norm(img1, img2):
    diff_img = torch.abs(img1 - img2)
    padded_img = torch.nn.functional.pad(diff_img, (1, 1, 1, 1), mode='replicate')
    diff_top = torch.abs(padded_img[:, :, 1:-1, 1:-1] - padded_img[:, :, :-2, 1:-1])
    diff_bottom = torch.abs(padded_img[:, :, 1:-1, 1:-1] - padded_img[:, :, 2:, 1:-1])
    diff_left = torch.abs(padded_img[:, :, 1:-1, 1:-1] - padded_img[:, :, 1:-1, :-2])
    diff_right = torch.abs(padded_img[:, :, 1:-1, 1:-1] - padded_img[:, :, 1:-1, 2:])
    pointwise_variation = (diff_top + diff_bottom + diff_left + diff_right) / 4.0
    tv_norm = torch.max(pointwise_variation.view(diff_img.size(0), -1), dim=1)[0]
    return tv_norm.item()

# Load CIFAR-10 dataset and apply basic transformation
transform = transforms.Compose([
    transforms.ToTensor()
])

cifar10_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Use a subset of CIFAR-10 for faster processing
dataloader = DataLoader(cifar10_train, batch_size=1, shuffle=True, num_workers=2)

# Initialize lists to store L2 and TV norm distances
l2_distances = []
tv_distances = []

# Loop through a subset of the data
translations = range(0, 21, 2)  # Different translation amounts

for i, (img, _) in enumerate(dataloader):
    if i == 10:  # Use first 10 images for this example
        break

    # img = img.cuda()  # Move to GPU if available

    for t in translations:
        img_translated = TF.affine(img, angle=0, translate=(t, t), scale=1, shear=0)  # Apply translation

        # Compute L2 norm and TV norm between original and translated image
        l2_distance = l2_norm(img, img_translated)
        tv_distance = total_variation_norm(img, img_translated)

        l2_distances.append(l2_distance)
        tv_distances.append(tv_distance)

# Plot the comparison of L2 norm and TV norm as translation increases
plt.figure(figsize=(10, 6))
plt.plot(translations, l2_distances[:len(translations)], label="L2 Norm Distance", marker='o')
plt.plot(translations, tv_distances[:len(translations)], label="TV Norm Distance", marker='x')
plt.xlabel("Translation Amount (pixels)")
plt.ylabel("Distance")
plt.title("Comparison of L2 Norm and TV Norm on Translated CIFAR-10 Images")
plt.legend()
plt.show()
