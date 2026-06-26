"""DP-Pix: pixelization with Laplace noise for differentially private image perturbation."""

import torch
import numpy as np
import cv2


# DP-Pix function: applies pixelization with Laplace noise for DP
def dp_pix(image, block_size, m, epsilon):
    img_np = np.transpose(image.cpu().numpy(), (1, 2, 0))  # Convert to numpy and (H, W, C)

    # Split into RGB channels
    b_channel, g_channel, r_channel = cv2.split(img_np)

    # Pixelization and Laplace noise for each channel
    new_b = pixelate_and_add_noise(b_channel, block_size, m, epsilon)
    new_g = pixelate_and_add_noise(g_channel, block_size, m, epsilon)
    new_r = pixelate_and_add_noise(r_channel, block_size, m, epsilon)

    # Merge back the channels
    dp_pix_image = cv2.merge((new_b, new_g, new_r))

    # Convert back to PyTorch tensor and transpose to (C, H, W)
    dp_pix_image = torch.tensor(np.transpose(dp_pix_image, (2, 0, 1)), dtype=torch.float32)

    return dp_pix_image

# Function to pixelate a single channel and add Laplace noise
def pixelate_and_add_noise(channel, block_size, m, epsilon):
    h, w = channel.shape
    new_channel = np.copy(channel)

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            # Average pixel value in the block
            block_mean = new_channel[i:i+block_size, j:j+block_size].mean()
            # Add Laplace noise
            noise = np.random.laplace(loc=0.0, scale=(1 * m) / (block_size**2 * epsilon))
            new_channel[i:i+block_size, j:j+block_size] = block_mean + noise

    return new_channel.clip(0, 1)  # Ensure values are within valid range
