import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


# Function to add Gaussian noise to an image
def add_gaussian_noise(image_tensor, mean=0, std=0.1):
    noise = torch.randn_like(image_tensor) * std + mean
    noisy_image = image_tensor + noise
    return torch.clamp(noisy_image, 0, 1)


# ROF TV denoising function
def denoise_rof(image, num_iter=100, lam=0.2, tau=0.125):
    u = image.clone()
    for i in range(num_iter):
        u_x = F.pad(u, (1, 1, 1, 1), mode='reflect')[:, :, 1:-1, 1:-1] - u
        u_y = F.pad(u, (1, 1, 1, 1), mode='reflect')[:, :, 1:-1, 1:-1] - u
        u_grad = torch.sqrt(u_x ** 2 + u_y ** 2 + 1e-6)
        u = u - tau * lam * u_grad
    return u


# Maximum Variation Metric denoising function
def denoise_mvm(image, num_iter=100, lam=0.2):
    u = image.clone()
    for i in range(num_iter):
        # Calculate Maximum Variation (as defined by the user)
        u_x = F.pad(u, (1, 1, 1, 1), mode='reflect')[:, :, 1:-1, 1:-1] - u
        u_y = F.pad(u, (1, 1, 1, 1), mode='reflect')[:, :, 1:-1, 1:-1] - u
        pointwise_variation = torch.max(torch.abs(u_x), torch.abs(u_y))
        u = u - lam * pointwise_variation
    return u


# Evaluation of the metrics (SSIM and PSNR)
def evaluate_metrics(original, denoised):
    original_np = original.squeeze().cpu().numpy().transpose(1, 2, 0)
    denoised_np = denoised.squeeze().cpu().numpy().transpose(1, 2, 0)
    img1_np = (original_np * 255).astype(np.uint8)
    img2_np = (denoised_np * 255).astype(np.uint8)
    ssim_val = ssim(img1_np, img2_np, channel_axis=2)
    psnr_val = psnr(original_np, denoised_np)
    return ssim_val, psnr_val


# Load and preprocess image
def load_image(image_path):
    image = Image.open(image_path)
    image = image.resize((128, 128))  # Resize for simplicity
    image_tensor = torch.tensor(np.array(image) / 255.0).permute(2, 0, 1).unsqueeze(0)
    return image_tensor


# Plot the results
def plot_results(original, noisy, denoised_tv, denoised_mvm):
    fig, axs = plt.subplots(1, 4, figsize=(15, 5))
    axs[0].imshow(original.squeeze().permute(1, 2, 0).cpu().numpy())
    axs[0].set_title("Original")
    axs[1].imshow(noisy.squeeze().permute(1, 2, 0).cpu().numpy())
    axs[1].set_title("Noisy")
    axs[2].imshow(denoised_tv.squeeze().permute(1, 2, 0).cpu().numpy())
    axs[2].set_title("Denoised TV")
    axs[3].imshow(denoised_mvm.squeeze().permute(1, 2, 0).cpu().numpy())
    axs[3].set_title("Denoised MVM")
    for ax in axs:
        ax.axis('off')
    plt.show()


# Main experiment
if __name__ == "__main__":
    # Load and add noise to the image
    image_tensor = load_image("../../data/BSDS300/images/train/8049.jpg")
    noisy_image = add_gaussian_noise(image_tensor)

    # Denoise using ROF TV and Maximum Variation Metric
    denoised_tv = denoise_rof(noisy_image)
    denoised_mvm = denoise_mvm(noisy_image)

    # Evaluate SSIM and PSNR for both methods
    ssim_tv, psnr_tv = evaluate_metrics(image_tensor, denoised_tv)
    ssim_mvm, psnr_mvm = evaluate_metrics(image_tensor, denoised_mvm)

    print(f"TV Denoising - SSIM: {ssim_tv:.4f}, PSNR: {psnr_tv:.2f}")
    print(f"MVM Denoising - SSIM: {ssim_mvm:.4f}, PSNR: {psnr_mvm:.2f}")

    # Plot the results
    plot_results(image_tensor, noisy_image, denoised_tv, denoised_mvm)
