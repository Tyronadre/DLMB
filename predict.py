import glob
import math
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from PIL import Image
from skimage.measure import label, regionprops
from torch.utils.data import DataLoader
from torchvision import transforms
from train import PatchDataset, UNet


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


def plot_predictions(model_name, model, data_name, dataloader, device, n_images=5, min_area=100, save_fig=True, plt_show=False):
    """
    Visualize n images from the dataloader. For each image, show:
      - Original image
      - Ground truth mask
      - Predicted mask
      - Difference map where:
            * Red pixels indicate extra predictions (false positives)
            * Blue pixels indicate missed predictions (false negatives)
      The IoU score is calculated and shown on the difference map.

    Parameters:
        model: The segmentation model.
        dataloader: A DataLoader providing (image, mask) pairs.
        device: torch.device to run predictions on.
        n_images: Number of images to visualize.
        min_area: (Unused in this function but can be used for cell filtering elsewhere.)
        save_fig: If True, save the resulting figure as a PNG file.
    """
    model.eval()  # Set model to evaluation mode.
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
            outputs = model(imgs)  # Assume raw logits, shape: [B, n_classes, H, W]
            preds = torch.argmax(outputs, dim=1)  # Predicted mask indices (assumes binary: 0 and 1)

            batch_size = imgs.size(0)
            for i in range(batch_size):
                if collected >= n_images:
                    break

                # Get the original image for plotting.
                orig_img = imgs[i].cpu().permute(1, 2, 0).numpy()
                # Get ground truth and predicted masks as 2D numpy arrays.
                true_mask = masks[i].cpu().numpy()
                pred_mask = preds[i].cpu().numpy()

                # Compute IoU score:
                # For binary segmentation, assume 0 = background, 1 = cell.
                TP = np.sum((true_mask == 1) & (pred_mask == 1))
                FP = np.sum((true_mask == 0) & (pred_mask == 1))
                FN = np.sum((true_mask == 1) & (pred_mask == 0))
                iou = TP / (TP + FP + FN + 1e-6)

                # Create a colored difference map.
                # Start with a black image.
                H, W = true_mask.shape
                diff_color = np.zeros((H, W, 3), dtype=np.uint8)
                # False positives: predicted cell (1) but ground truth is background (0) --> Red.
                fp_mask = np.logical_and(true_mask == 0, pred_mask == 1)
                diff_color[fp_mask] = [255, 0, 0]  # Red.
                # False negatives: ground truth cell (1) but predicted background (0) --> Blue.
                fn_mask = np.logical_and(true_mask == 1, pred_mask == 0)
                diff_color[fn_mask] = [0, 0, 255]  # Blue.

                # Plot the four panels.
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
                ax_pred.set_title("Predicted Mask")
                ax_pred.axis("off")

                # Difference map with IoU score.
                ax_diff.imshow(diff_color)
                ax_diff.set_title(f"Diff Map\nIoU: {iou:.3f}")
                ax_diff.axis("off")

                collected += 1

            if collected >= n_images:
                break

    plt.tight_layout()

    # set the title of the plot with the size of the patches
    plt.suptitle("Image Segmentation Predictions " + model_name + ". Size: " + str(H) + "x" + str(W))
    if save_fig:
        os.makedirs("predictions",exist_ok=True)
        plt.savefig(f"predictions/{model_name}_predictions_with_diff_{data_name}.png")
    if plt_show:
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


def predict_full_image(model, image, patch_size=256, stride=256, device=None):
    """
    Predicts a full segmentation mask for a whole image using a patch-based sliding window approach.

    Parameters:
      - model: the segmentation model (trained on patches).
      - image: the full image to segment. Can be a PIL.Image, a NumPy array, or a pre-converted tensor.
      - patch_size: the size (in pixels) of each square patch.
      - stride: stride of the sliding window. If stride < patch_size, overlapping patches will be averaged.
      - device: torch.device on which to run predictions (if None, uses CUDA if available).

    Returns:
      - predicted_mask: a 2D NumPy array with the predicted class for each pixel.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()

    # Convert input image to a tensor.
    if isinstance(image, np.ndarray):
        # Assume image is H x W x C with values [0, 255]
        img_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    elif isinstance(image, Image.Image):
        img_tensor = transforms.ToTensor()(image).unsqueeze(0)
    elif isinstance(image, torch.Tensor):
        # If already tensor, ensure shape is (1, C, H, W)
        img_tensor = image if image.ndim == 4 else image.unsqueeze(0)
    else:
        raise TypeError("Unsupported image type")

    img_tensor = img_tensor.to(device)
    _, C, H, W = img_tensor.shape

    # Pad the image so that its dimensions are multiples of patch_size.
    new_H = math.ceil(H / patch_size) * patch_size
    new_W = math.ceil(W / patch_size) * patch_size
    pad_bottom = new_H - H
    pad_right = new_W - W
    img_tensor = F.pad(img_tensor, (0, pad_right, 0, pad_bottom), mode='reflect')
    _, _, H_pad, W_pad = img_tensor.shape

    # Assume model output has n_classes channels.
    n_classes = 2  # Adjust if needed.
    output_tensor = torch.zeros((1, n_classes, H_pad, W_pad), device=device)
    count_tensor = torch.zeros((1, n_classes, H_pad, W_pad), device=device)

    # Slide a window over the padded image.
    for i in range(0, H_pad, stride):
        for j in range(0, W_pad, stride):
            patch = img_tensor[:, :, i:i + patch_size, j:j + patch_size]
            # Pad patch if it is smaller than patch_size.
            if patch.shape[2] != patch_size or patch.shape[3] != patch_size:
                pad_h = patch_size - patch.shape[2]
                pad_w = patch_size - patch.shape[3]
                patch = F.pad(patch, (0, pad_w, 0, pad_h), mode='reflect')
            with torch.no_grad():
                patch_output = model(patch)  # shape: (1, n_classes, patch_size, patch_size)
            output_tensor[:, :, i:i + patch_size, j:j + patch_size] += patch_output
            count_tensor[:, :, i:i + patch_size, j:j + patch_size] += 1

    # Average overlapping predictions.
    output_tensor /= count_tensor

    # Crop to original size.
    output_tensor = output_tensor[:, :, :H, :W]
    predicted_mask = torch.argmax(output_tensor, dim=1).squeeze(0).cpu().numpy()

    return predicted_mask


def plot_full_image_predictions_n(model_name, model, image_paths, mask_paths, device, patch_size=256, stride=256, n_images=5,
                                  save_fig=True, plt_show=False):
    """
    For each of the provided full images, predict the segmentation using a sliding window approach,
    then plot a row with:
      - Original image
      - Ground Truth mask
      - Predicted mask
      - Colored Difference Map:
          * Red indicates false positives (extra predictions)
          * Blue indicates false negatives (missed predictions)
      The IoU score is calculated and displayed on the difference map.

    Parameters:
      - model: the segmentation model.
      - image_paths: list of file paths to full images.
      - mask_paths: list of file paths to corresponding ground truth masks.
      - device: torch.device on which to run prediction.
      - patch_size: size of patches used in sliding window.
      - stride: stride for sliding window.
      - n_images: number of images to visualize.
      - save_fig: if True, the figure is saved as a PNG file.
    """
    model.eval()

    # Create a figure with n_images rows and 4 columns.
    fig, axs = plt.subplots(n_images, 4, figsize=(20, 5 * n_images))
    if n_images == 1:
        axs = np.expand_dims(axs, 0)

    for idx in range(n_images):
        # Load original full image.
        orig_img = Image.open(image_paths[idx]).convert("RGB")
        # Load ground truth mask.
        # Here we assume the ground truth mask is a grayscale image with labels (e.g., 0 and 1).
        gt_mask = Image.open(mask_paths[idx]).convert("L")
        gt_mask_np = np.array(gt_mask)

        # Predict the segmentation for the full image.
        pred_mask = predict_full_image(model, orig_img, patch_size=patch_size, stride=stride, device=device)

        # Compute IoU (for binary segmentation: label 1 is foreground).
        TP = np.sum((gt_mask_np == 1) & (pred_mask == 1))
        FP = np.sum((gt_mask_np == 0) & (pred_mask == 1))
        FN = np.sum((gt_mask_np == 1) & (pred_mask == 0))
        iou = TP / (TP + FP + FN + 1e-6)

        # Create a colored difference map.
        H, W = gt_mask_np.shape
        diff_color = np.zeros((H, W, 3), dtype=np.uint8)
        # False positives (predicted cell but GT is background): red.
        fp = np.logical_and(gt_mask_np == 0, pred_mask == 1)
        diff_color[fp] = [255, 0, 0]
        # False negatives (GT has cell but prediction is background): blue.
        fn = np.logical_and(gt_mask_np == 1, pred_mask == 0)
        diff_color[fn] = [0, 0, 255]

        # Plotting: each row shows Original, GT, Prediction, and Difference Map.
        ax_orig = axs[idx, 0]
        ax_gt = axs[idx, 1]
        ax_pred = axs[idx, 2]
        ax_diff = axs[idx, 3]

        # Original image.
        ax_orig.imshow(orig_img)
        ax_orig.set_title("Original Image")
        ax_orig.axis("off")

        # Ground Truth Mask.
        ax_gt.imshow(gt_mask_np, cmap='gray')
        ax_gt.set_title("Ground Truth Mask")
        ax_gt.axis("off")

        # Predicted Mask.
        ax_pred.imshow(pred_mask, cmap='gray')
        ax_pred.set_title("Predicted Mask")
        ax_pred.axis("off")

        # Difference Map with IoU score.
        ax_diff.imshow(diff_color)
        ax_diff.set_title(f"Difference Map\nIoU: {iou:.3f}")
        ax_diff.axis("off")

    plt.tight_layout()


    plt.suptitle("Full Image Predictions")

    if save_fig:
        os.makedirs("predictions",exist_ok=True)
        plt.savefig(f"predictions/{model_name}_full_image_predictions.png")
    if plt_show:
        plt.show()


def main():
    # Directories for the test patches.
    test_image_dir = "E:/data/test"
    test_mask_dir = "E:/data/test_masks"
    patch_100_image = "E:/data/patches_100/test_images"
    patch_100_mask = "E:/data/patches_100/test_masks"
    patch_128_image = "E:/data/patches_128/test_images"
    patch_128_mask = "E:/data/patches_128/test_masks"
    patch_256_image = "E:/data/patches_256/test_images"
    patch_256_mask = "E:/data/patches_256/test_masks"

    transform = transforms.Compose([transforms.ToTensor()])
    print("creating dataloader")

    # Create the test dataset and loader.
    loader_128 = DataLoader(PatchDataset(patch_128_image, patch_128_mask, transform=transform), batch_size=4,
                            shuffle=False)
    loader_256 = DataLoader(PatchDataset(patch_256_image, patch_256_mask, transform=transform), batch_size=4,
                            shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_path = "E:\\ai_models"

    for model_name in os.listdir(models_path):
        model_path = os.path.join(models_path, model_name)
        if not model_path.endswith(".pth"):
            continue
        try:
            if os.path.exists(model_path):
                model_data = model_name.split("_")
                model = UNet(in_channels=3, out_channels=2, depth=int(model_data[1]),
                             base_filters=int(model_data[3]))

                model.load_state_dict(torch.load(model_path, map_location=device))
                print("Model " + model_path + " loaded successfully.")
                model.to(device)

                image_paths = sorted(
                    [os.path.join(test_image_dir, f) for f in os.listdir(test_image_dir) if
                     f.lower().endswith('.tif')])
                mask_paths = sorted(
                    [os.path.join(test_mask_dir, f) for f in os.listdir(test_mask_dir) if
                     f.lower().endswith('.tif')])

                n_images_to_show = min(5, len(image_paths))
                print("Predicting full Image")
                plot_full_image_predictions_n(model_name, model, image_paths, mask_paths, device, patch_size=256,
                                              stride=256,
                                              n_images=n_images_to_show, save_fig=True)

                for data_name, test_loader in {"patch-128": loader_128, "patch-256": loader_256}.items():
                    try:
                        print(f"Predicting {data_name}")
                        plot_predictions(model_name, model, data_name, test_loader, device, n_images=3, save_fig=True)


                    except Exception as e:
                        print(f"Error plotting predictions: {e}")
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")


if __name__ == '__main__':
    main()
