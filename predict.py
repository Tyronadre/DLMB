import glob
import os
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from train import PatchDataset, UNet

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


def plot_predictions(model_name, model, dataloader, device, n_images=5, save_fig=False, min_area=100):
    model.eval()
    collected = 0
    fig, axs = plt.subplots(n_images, 4, figsize=(20, 5 * n_images))
    plt.suptitle(f"model: {model_name}")

    if n_images == 1:
        axs = np.expand_dims(axs, 0)

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)

            # Process each image in the batch.
            batch_size = images.size(0)
            for i in range(batch_size):
                if collected >= n_images:
                    break

                orig_img = images[i].cpu().permute(1, 2, 0).numpy()
                true_mask = masks[i].cpu().numpy()
                pred_mask = predictions[i].cpu().numpy()
                binary_pred = (pred_mask > 0).astype(np.uint8)
                cell_count = count_cells_in_patch(binary_pred, min_area=min_area)
                diff_map = (true_mask != pred_mask).astype(np.uint8)
                ax_orig = axs[collected, 0]
                ax_gt = axs[collected, 1]
                ax_pred = axs[collected, 2]
                ax_diff = axs[collected, 3]

                ax_orig.imshow(orig_img)
                ax_orig.set_title("Original Image")
                ax_orig.axis("off")

                ax_gt.imshow(true_mask, cmap='gray')
                ax_gt.set_title("Ground Truth Mask")
                ax_gt.axis("off")

                ax_pred.imshow(pred_mask, cmap='gray')
                title = f"Predicted Mask"
                # title += f"\nCell Count (min_area={min_area}): {cell_count}"
                ax_pred.set_title(title)
                ax_pred.axis("off")

                ax_diff.imshow(diff_map, cmap='gray')
                ax_diff.set_title("Difference Map")
                ax_diff.axis("off")

                collected += 1

            if collected >= n_images:
                break

    plt.tight_layout()
    if save_fig:
        if not os.path.exists("saved_pred"):
            os.makedirs("saved_pred")
        plt.savefig(f"saved_pred/predictions_{model_name}.png")
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
    # Directories for the test patches.
    test_patch_image_dir = "data/npm_img"
    test_patch_mask_dir = "data/npm_mask"

    transform = transforms.Compose([transforms.ToTensor()])

    # Create the test dataset and loader.
    test_dataset = PatchDataset(test_patch_image_dir, test_patch_mask_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    # Set device and load the saved model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_path = "saved_models"

    for model_name in os.listdir(models_path):
        model_path = os.path.join(models_path, model_name)
        try:
            if os.path.exists(model_path):
                # if not model_path.__contains__("d_6"): continue
                model_data = model_name.split("_")
                model = UNet(in_channels=3, out_channels=2, depth=int(model_data[1]), base_filters=int(model_data[3]))

                model.load_state_dict(torch.load(model_path, map_location=device))
                print("Model " + model_path + " loaded successfully.")
                model.to(device)
                plot_predictions(model_name, model, test_loader, device, n_images=3, save_fig=True)
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")


if __name__ == '__main__':
    main()
