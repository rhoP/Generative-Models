import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from PIL import Image
import os
import io


# Function to compute Total Variation Norm
def total_variation_norm(img):
    padded_img = F.pad(img, (1, 1, 1, 1), mode='replicate')
    diff_top = torch.abs(padded_img[:, :, 1:-1, 1:-1] - padded_img[:, :, :-2, 1:-1])
    diff_bottom = torch.abs(padded_img[:, :, 1:-1, 1:-1] - padded_img[:, :, 2:, 1:-1])
    diff_left = torch.abs(padded_img[:, :, 1:-1, 1:-1] - padded_img[:, :, 1:-1, :-2])
    diff_right = torch.abs(padded_img[:, :, 1:-1, 1:-1] - padded_img[:, :, 1:-1, 2:])
    pointwise_variation = (diff_top + diff_bottom + diff_left + diff_right) / 4.0
    tv_norm = torch.max(pointwise_variation.view(img.size(0), -1), dim=1)[0]
    return tv_norm.item()

def total_variation_dist(img1, img2):
    diff_img = torch.abs(img1 - img2)
    padded_img = F.pad(diff_img, (1, 1, 1, 1), mode='replicate')
    diff_top = torch.abs(padded_img[:, :, 1:-1, 1:-1] - padded_img[:, :, :-2, 1:-1])
    diff_bottom = torch.abs(padded_img[:, :, 1:-1, 1:-1] - padded_img[:, :, 2:, 1:-1])
    diff_left = torch.abs(padded_img[:, :, 1:-1, 1:-1] - padded_img[:, :, 1:-1, :-2])
    diff_right = torch.abs(padded_img[:, :, 1:-1, 1:-1] - padded_img[:, :, 1:-1, 2:])
    pointwise_variation = (diff_top + diff_bottom + diff_left + diff_right) / 4.0
    tv_norm = torch.max(pointwise_variation.view(diff_img.size(0), -1), dim=1)[0]
    return tv_norm.item()


# Function to apply JPEG compression and return compressed image
def compress_image(image_tensor, quality):
    img_pil = TF.to_pil_image(image_tensor.cpu())
    buffer = io.BytesIO()
    img_pil.save(buffer, format="JPEG", quality=quality)
    compressed_img_pil = Image.open(buffer)
    compressed_img_tensor = TF.to_tensor(compressed_img_pil)
    return compressed_img_tensor


# Load and transform image
image_path = '../../data/Einstein/diffs/'
image_filenames = [f for f in os.listdir(image_path) if f.endswith('.jpg') or f.endswith('.gif')]
main_image = TF.to_tensor(Image.open('../../data/Einstein/einstein.gif')).unsqueeze(0)
images = [Image.open(os.path.join(image_path, f)) for f in image_filenames]
image_tensors = [TF.to_tensor(i).unsqueeze(0) for i in images]


tvs = []

for diff in image_tensors:
    tvs.append(total_variation_dist(main_image, diff))

# Print out TV norms
for q, tv in zip(image_filenames, tvs):
    print(f"TV after JPEG compression (quality {q}): {tv}")
