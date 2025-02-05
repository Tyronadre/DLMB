import glob
import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from train import PatchDataset


# Define your U-Net (or import if it's defined elsewhere)
class DoubleConv(nn.Module):
    """(Convolution => BatchNorm => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2):
        super(UNet, self).__init__()

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 512))

        self.up1 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(1024, 256)
        self.up2 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(512, 128)
        self.up3 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(256, 64)
        self.up4 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(128, 64)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)  # [B, 64, H, W]
        x2 = self.down1(x1)  # [B, 128, H/2, W/2]
        x3 = self.down2(x2)  # [B, 256, H/4, W/4]
        x4 = self.down3(x3)  # [B, 512, H/8, W/8]
        x5 = self.down4(x4)  # [B, 512, H/16, W/16]

        x = self.up1(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.conv1(x)

        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.conv2(x)

        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv3(x)

        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv4(x)

        logits = self.outc(x)
        return logits


import torch
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label, regionprops


def count_cells_in_patch(mask, min_area=50):
    """
    Count cells in a binary mask using connected components and a minimum area threshold.

    Parameters:
        mask (np.array): A 2D binary numpy array where cell pixels have value 1.
        min_area (int): The minimum area (in pixels) for a region to be considered a valid cell.

    Returns:
        cell_count (int): Number of cells with an area >= min_area.
    """
    # Label connected regions in the binary mask.
    labeled = label(mask)
    cell_count = 0

    for region in regionprops(labeled):
        # Only count regions with area above the minimum threshold.
        if region.area >= min_area:
            cell_count += 1

    return cell_count

def plot_predictions(model, dataloader, device, n_images=5, save_fig=False, min_area=100):
    model.eval()  # Set model to evaluation mode
    collected = 0

    # Create a figure with n_images rows and 4 columns.
    fig, axs = plt.subplots(n_images, 4, figsize=(20, 5 * n_images))

    # Ensure axs is 2D (if n_images == 1, force it to be a 2D array).
    if n_images == 1:
        axs = np.expand_dims(axs, 0)

    with torch.no_grad():
        for imgs, masks in dataloader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            outputs = model(imgs)  # Raw logits, shape: [B, n_classes, H, W]
            preds = torch.argmax(outputs, dim=1)  # Predicted mask indices

            # Process each image in the batch.
            batch_size = imgs.size(0)
            for i in range(batch_size):
                if collected >= n_images:
                    break

                # Convert the original image from (C,H,W) to (H,W,C) for plotting.
                orig_img = imgs[i].cpu().permute(1, 2, 0).numpy()
                # Get ground truth and predicted masks as numpy arrays.
                true_mask = masks[i].cpu().numpy()
                pred_mask = preds[i].cpu().numpy()

                # Create a binary mask from the predicted mask.
                # Adjust the threshold or condition if you have multi-class values.
                binary_pred = (pred_mask > 0).astype(np.uint8)
                cell_count = count_cells_in_patch(binary_pred, min_area=min_area)

                # Compute the difference map (1 indicates a disagreement).
                diff_map = (true_mask != pred_mask).astype(np.uint8)

                # Plot the four panels.
                ax_orig = axs[collected, 0]
                ax_gt = axs[collected, 1]
                ax_pred = axs[collected, 2]
                ax_diff = axs[collected, 3]

                # Original image.
                ax_orig.imshow(orig_img)
                ax_orig.set_title("Original Image")
                ax_orig.axis("off")

                # Ground truth mask (grayscale).
                ax_gt.imshow(true_mask, cmap='gray')
                ax_gt.set_title("Ground Truth Mask")
                ax_gt.axis("off")

                # Predicted mask with cell count.
                ax_pred.imshow(pred_mask, cmap='gray')
                title = f"Predicted Mask\nCell Count (min_area={min_area}): {cell_count}"
                ax_pred.set_title(title)
                ax_pred.axis("off")

                # Difference map.
                ax_diff.imshow(diff_map, cmap='gray')
                ax_diff.set_title("Difference Map")
                ax_diff.axis("off")

                collected += 1

            if collected >= n_images:
                break

    plt.tight_layout()
    if save_fig:
        plt.savefig("predictions_with_cell_counts.png")
    plt.show()

def show_sample_patches(image_patch_dir, mask_patch_dir, num_samples=6):
    """
    Load and display a few image patches and their corresponding masks.
    """
    image_files = sorted(glob.glob(os.path.join(image_patch_dir, "*.npy")))
    mask_files = sorted(glob.glob(os.path.join(mask_patch_dir, "*.npy")))

    num_samples = min(num_samples, len(image_files))

    fig, axs = plt.subplots(num_samples, 2, figsize=(8, 4 * num_samples))

    for i in range(num_samples):
        # Load image and mask patches from npy files
        img_patch = np.load(image_files[i])
        mask_patch = np.load(mask_files[i])

        # Convert image patch to displayable format
        # (Note: mask_patch is assumed to be grayscale)
        axs[i, 0].imshow(img_patch.astype('uint8'))
        axs[i, 0].set_title("Image Patch")
        axs[i, 0].axis("off")

        # For the mask, squeeze the extra channel if present.
        if mask_patch.ndim == 3 and mask_patch.shape[-1] == 1:
            mask_patch = mask_patch.squeeze(-1)
        axs[i, 1].imshow(mask_patch, cmap='gray')
        axs[i, 1].set_title("Mask Patch")
        axs[i, 1].axis("off")

    plt.tight_layout()
    plt.show()

# ---------------------------
# Main Execution for Prediction Plotting
# ---------------------------

def main():
    train_patch_image_dir = "data/patches/train_images"
    train_patch_mask_dir = "data/patches/train_masks"
    # show_sample_patches(train_patch_image_dir, train_patch_mask_dir, num_samples=10)


    # Directories for the test patches.
    test_patch_image_dir = "data/patches/test_images"
    test_patch_mask_dir = "data/patches/test_masks"

    # Define a basic transform (should match what was used in training).
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Add normalization if you used it during training.
    ])

    # Create the test dataset and loader.
    test_dataset = PatchDataset(test_patch_image_dir, test_patch_mask_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)

    # Set device and load the saved model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(n_channels=3, n_classes=2)
    model_path = "saved_models/1.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded successfully.")
    else:
        print(f"Model path {model_path} does not exist.")
        return

    model.to(device)

    plot_predictions(model, test_loader, device, n_images=9, save_fig=True)


if __name__ == '__main__':
    main()
