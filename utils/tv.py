import numpy as np


def tv(img):
    """
    Computes the total variation norm of a 2D image
    :param img: 2D numpy array
    :return: float
    """

    n, m = img.shape
    local_variation = np.zeros((n, m), dtype=float)
    for i in range(1, n):
        for j in range(1, m):
            neighbors = []
            if i>0:
                neighbors.append(img[i-1, j])
            if i < n-1:
                neighbors.append(img[i+1, j])
            if j > 0:
                neighbors.append(img[i, j-1])
            if j < m-1:
                neighbors.append(img[i, j+1])

            if neighbors:
                local_variation[i, j] = np.mean([abs(img[i, j] - neighbor) for neighbor in neighbors])

    return np.max(local_variation)


def batch_total_variation_norm(images):
    """
    Compute the total variation (TV) norm for a batch of 2D images.
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