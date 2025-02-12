import glob
import math
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import xml.etree.ElementTree as ET

from PIL import Image
from skimage.measure import label, regionprops
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

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


def plot_predictions(model_name, model, data_name, dataloader, device, n_images=5, min_area=100, save_fig=True,
                     plt_show=False):
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
        os.makedirs("predictions_256", exist_ok=True)
        plt.savefig(f"predictions_256/{model_name}_predictions_with_diff_{data_name}.png")
        print(f"Predictions saved to predictions/{model_name}_predictions_with_diff_{data_name}.png")
    if plt_show:
        plt.show()


def plot_prediction_with_cell_counts(model_name, model, dataloader, device, annotate_color='red', font_size=12, min_area=100):
    """
    Loads one image from the dataloader, predicts its segmentation mask with the model,
    counts the cells in the predicted binary mask, overlays cell numbers on the predicted image,
    and displays a two-panel plot comparing the ground truth mask (left) to the predicted mask with annotations (right).

    Parameters:
        model_name: Name of the model.
        model: Trained segmentation model.
        dataloader: PyTorch DataLoader yielding (image, mask) pairs.
        device: torch.device for running the model.
        annotate_color (str): Color for cell number annotations.
        font_size (int): Font size for annotations.
        min_area (int): Minimum area for a region to be considered a cell.

    Returns:
        fig: The matplotlib figure.
    """
    model.eval()

    # Get one batch from the dataloader and pick the first image & mask.
    for imgs, masks in dataloader:
        # Use the first sample from the batch.
        img = imgs[0]  # shape: (C, H, W)
        gt_mask = masks[0]  # shape: (H, W) or (1, H, W)
        break

    # Move image to device and add batch dimension.
    img = img.unsqueeze(0).to(device)

    # Predict the mask.
    with torch.no_grad():
        output = model(img)  # Assume shape: (1, n_classes, H, W)
    # For binary segmentation, take argmax along channel dimension.
    pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()  # shape: (H, W)

    # Convert ground truth mask to numpy.
    if gt_mask.ndim == 3:
        gt_mask = gt_mask.squeeze(0)
    gt_mask_np = gt_mask.cpu().numpy() if torch.is_tensor(gt_mask) else np.array(gt_mask)

    # Create a color version of the predicted mask for annotation.
    # Here we multiply the binary mask by 255 and stack it into 3 channels.
    pred_color = np.stack([pred_mask * 255] * 3, axis=-1).astype(np.uint8)

    # Label connected components (cells) in the predicted mask.
    labeled = label(pred_mask)
    regions = regionprops(labeled)
    for region in regions:
        if region.area < min_area:
            regions.remove(region)

    # Create a plot with 2 panels.
    fig, axs = plt.subplots(1, 2, figsize=(14, 7))

    # Left panel: display the ground truth mask.
    axs[0].imshow(gt_mask_np, cmap='gray')
    axs[0].set_title("Ground Truth Mask")
    axs[0].axis("off")

    # Right panel: display the predicted mask (color) with cell numbers overlaid.
    axs[1].imshow(pred_color)
    axs[1].set_title("Predicted Mask with Cell Numbers with min area {}".format(min_area))
    axs[1].axis("off")

    # Annotate each cell with its number at the centroid.
    for idx, region in enumerate(regions, start=1):
        y, x = region.centroid  # (row, col)
        axs[1].text(x, y, str(idx), color=annotate_color, fontsize=font_size,
                    fontweight='bold', ha='center', va='center')

    total_cells = len(regions)
    fig.suptitle(f"Total Cells Detected: {total_cells} ({model_name})", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs("predictions_256", exist_ok=True)
    plt.savefig(f"predictions_256/{model_name}_cell_count.png")
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


def plot_full_image_predictions_n(model_name, model, image_paths, mask_paths, device, patch_size=256, stride=256,
                                  n_images=5,
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
        os.makedirs("predictions_256", exist_ok=True)
        plt.savefig(f"predictions_256/{model_name}_full_image_predictions.png")
        print(f"Predictions saved to predictions/{model_name}_full_image_predictions.png")
    if plt_show:
        plt.show()


def compute_iou(pred, gt):
    """
    Compute the Intersection-over-Union (IoU) between two binary masks.
    Both pred and gt should be numpy arrays with values 0 and 1.
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return intersection / (union + 1e-6)


def compute_dice(pred, gt):
    """
    Compute the Dice coefficient between two binary masks.
    Both pred and gt should be numpy arrays with values 0 and 1.
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    return (2 * intersection) / (pred.sum() + gt.sum() + 1e-6)


def evaluate_patch_loader(model, loader, device):
    """
    Evaluate the model on a DataLoader that provides patch predictions.
    Returns the average IoU and Dice over all patches.
    """
    iou_list = []
    dice_list = []
    model.eval()
    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            outputs = model(imgs)  # Assume outputs shape: (B, n_classes, H, W)
            preds = torch.argmax(outputs, dim=1)  # shape: (B, H, W)
            preds_np = preds.cpu().numpy()
            masks_np = masks.cpu().numpy()
            for pred, gt in zip(preds_np, masks_np):
                # Ensure binary (assuming foreground label is 1)
                iou_list.append(compute_iou((pred > 0).astype(np.uint8), (gt > 0).astype(np.uint8)))
                dice_list.append(compute_dice((pred > 0).astype(np.uint8), (gt > 0).astype(np.uint8)))
    model.train()
    return np.mean(iou_list), np.mean(dice_list)


def evaluate_full_images(model, image_paths, mask_paths, device, patch_size=256, stride=256):
    """
    Evaluate the model on full test images using the sliding-window approach.
    Returns the average IoU and Dice over all test images.
    """
    iou_list = []
    dice_list = []
    for img_path, mask_path in zip(image_paths, mask_paths):
        image = Image.open(img_path).convert("RGB")
        gt_mask = Image.open(mask_path).convert("L")
        gt_np = np.array(gt_mask)
        pred_np = predict_full_image(model, image, patch_size=patch_size, stride=stride, device=device)
        # Convert to binary masks (assuming label 1 is foreground)
        iou_list.append(compute_iou((pred_np > 0).astype(np.uint8), (gt_np > 0).astype(np.uint8)))
        dice_list.append(compute_dice((pred_np > 0).astype(np.uint8), (gt_np > 0).astype(np.uint8)))
    return np.mean(iou_list), np.mean(dice_list)


def plot_iou_dice_curves(folder="saved_models"):
    # Directories for the test images and patches.
    test_image_dir = "data/test"
    test_mask_dir = "data/test_masks"
    patch_128_image = "data/patches_128/test_images"
    patch_128_mask = "data/patches_128/test_masks"
    patch_256_image = "data/patches_256/test_images"
    patch_256_mask = "data/patches_256/test_masks"

    transform = transforms.Compose([transforms.ToTensor()])
    print("Creating dataloaders...")

    # Create DataLoaders for patch evaluations.
    loader_128 = DataLoader(PatchDataset(patch_128_image, patch_128_mask, transform=transform), batch_size=4,
                            shuffle=False)
    loader_256 = DataLoader(PatchDataset(patch_256_image, patch_256_mask, transform=transform), batch_size=4,
                            shuffle=False)

    device = torch.device("cuda")
    models_path = "saved_models_256"

    # We will store results in a dictionary.
    # For each model we record metrics for three cases: 'patch_128', 'patch_256', 'full'
    results = {}

    # Get full image file paths.
    image_paths = sorted(
        [os.path.join(test_image_dir, f) for f in os.listdir(test_image_dir) if f.lower().endswith('.tif')])
    mask_paths = sorted(
        [os.path.join(test_mask_dir, f) for f in os.listdir(test_mask_dir) if f.lower().endswith('.tif')])

    for model_name in os.listdir(models_path):
        model_path = os.path.join(models_path, model_name)
        if not model_path.endswith(".pth"):
            continue
        try:
            if os.path.exists(model_path):
                # Parse model parameters from the filename.
                # For example, if model_name is in the form: "d_4_f_64_lr_0.0001.pth"
                model_data = model_name.split("_")
                depth = int(model_data[1])
                base_filters = int(model_data[3])
                model = UNet(in_channels=3, out_channels=2, depth=depth, base_filters=base_filters)
                model.load_state_dict(torch.load(model_path, map_location=device))
                print("Model " + model_path + " loaded successfully.")
                model.to(device)

                # Evaluate on 128 patches.
                iou_128, dice_128 = evaluate_patch_loader(model, loader_128, device)
                # Evaluate on 256 patches.
                iou_256, dice_256 = evaluate_patch_loader(model, loader_256, device)
                # Evaluate on full images.
                iou_full, dice_full = evaluate_full_images(model, image_paths, mask_paths, device, patch_size=256,
                                                           stride=256)

                results[model_name] = {
                    "patch_128": {"iou": iou_128, "dice": dice_128},
                    "patch_256": {"iou": iou_256, "dice": dice_256},
                    "full": {"iou": iou_full, "dice": dice_full}
                }
        except Exception as e:
            print(f"Error processing model {model_name}: {e}")

    # Create grouped bar charts to compare IoU and Dice for each model across the three settings.
    model_names = list(results.keys())
    datasets = ["patch_128", "patch_256", "full"]
    # Create an x-axis position for each model.
    x = np.arange(len(model_names))
    width = 0.25  # width of each bar

    # Create one figure with two subplots (side-by-side).
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Plot IoU ---
    for idx, ds in enumerate(datasets):
        # For each dataset (test setting), extract the IoU scores for each model.
        ious = [results[m][ds]["iou"] for m in model_names]
        axes[0].bar(x + idx * width, ious, width, label=ds)
    axes[0].set_ylabel("IoU")
    axes[0].set_title("IoU Comparison Across Models and Test Settings")
    axes[0].set_xticks(x + width)
    axes[0].set_xticklabels(model_names, rotation=45)
    axes[0].legend()

    # --- Plot Dice Score ---
    for idx, ds in enumerate(datasets):
        dices = [results[m][ds]["dice"] for m in model_names]
        axes[1].bar(x + idx * width, dices, width, label=ds)
    axes[1].set_ylabel("Dice Score")
    axes[1].set_title("Dice Score Comparison Across Models and Test Settings")
    axes[1].set_xticks(x + width)
    axes[1].set_xticklabels(model_names, rotation=45)
    axes[1].legend()

    plt.suptitle("Evaluation Metrics for Test Images (Patches & Full Image)")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    ## Save the plot
    os.makedirs("predictions", exist_ok=True)
    plt.savefig(f"predictions/iou_dice_curves.png")


def extract_actual_cell_count(xml_path):
    """
    Given an XML file (for a test image) that contains cell regions (each defined by a <region> tag),
    returns the number of cells (regions) specified.

    Modify the tag or XPath below if your XML structure is different.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        regions = root.findall('.//Region')
        return len(regions)
    except Exception as e:
        print(f"Error parsing XML {xml_path}: {e}")
        return 0


def predict_cell_count(model, image, device, patch_size=256, stride=256, min_size=100):
    """
    Given a full image and a model, predict its segmentation mask and count the cells
    (i.e. connected components) in the predicted binary mask.

    Returns the cell count (an integer).
    """
    pred_mask = predict_full_image(model, image, patch_size, stride, device)
    labeled_mask = label(pred_mask)
    regions = regionprops(labeled_mask)
    for region in regions:
        if region.area < min_size:
            regions.remove(region)
    return len(regions)


def evaluate_cell_count_for_models(models_path, test_image_dir, device, patch_size=256, stride=256):
    """
    For each model in models_path, iterate over all test images in test_image_dir.
    For each image, read its corresponding XML file (assumed to have the same basename as the image)
    to get the actual cell count and predict the cell count using the model.
    Then, average the counts over all test images.

    Returns a dictionary with keys = model filenames and values = {"predicted_avg": X, "actual_avg": Y}.
    """
    results = {}
    image_files = sorted([f for f in os.listdir(test_image_dir) if f.lower().endswith('.tif')])

    for model_name in os.listdir(models_path):
        model_path = os.path.join(models_path, model_name)
        if not model_path.endswith(".pth"):
            continue
        try:
            # Parse parameters from filename, e.g. "d_4_f_64_lr_0.0001.pth"
            model_data = model_name.split("_")
            depth = int(model_data[1])
            base_filters = int(model_data[3])
            # Initialize and load model (replace UNet with your actual model definition).
            model = UNet(in_channels=3, out_channels=2, depth=depth, base_filters=base_filters)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            continue

        pred_counts = []
        actual_counts = []
        for img_file in tqdm(image_files, f"Predicting {model_name}"):
            img_path = os.path.join(test_image_dir, img_file)
            image = Image.open(img_path).convert("RGB")

            # Predict cell count.
            pred_count = predict_cell_count(model, image, device, patch_size, stride)
            pred_counts.append(pred_count)

            # Get actual cell count from XML.
            base = os.path.splitext(img_file)[0]
            xml_path = os.path.join(test_image_dir, base + ".xml")
            if not os.path.exists(xml_path):
                print(f"XML file {xml_path} not found; skipping actual count for {img_file}.")
                continue
            actual_count = extract_actual_cell_count(xml_path)
            actual_counts.append(actual_count)

        # Compute averages (if there are valid actual counts).
        if len(pred_counts) > 0:
            avg_pred = np.mean(pred_counts)
        else:
            avg_pred = 0
        if len(actual_counts) > 0:
            avg_actual = np.mean(actual_counts)
        else:
            avg_actual = 0

        results[model_name] = {"predicted_avg": avg_pred, "actual_avg": avg_actual}
        print(f"Model {model_name}: Avg. Predicted = {avg_pred:.2f}, Avg. Actual = {avg_actual:.2f}")
    return results


def plot_cell_count_comparison(results):
    """
    Given a results dictionary (keys: model names, values: {"predicted_avg": X, "actual_avg": Y}),
    plot a grouped bar chart comparing the average predicted cell count vs the average actual cell count.
    """
    model_names = list(results.keys())
    predicted_avgs = [results[m]["predicted_avg"] for m in model_names]
    actual_avgs = [results[m]["actual_avg"] for m in model_names]

    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width / 2, predicted_avgs, width, label='Predicted Average')
    ax.bar(x + width / 2, actual_avgs, width, label='Actual Average')
    ax.set_ylabel("Average Cell Count")
    ax.set_title("Average Predicted vs. Actual Cell Counts Across Models")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    plt.tight_layout()
    plt.show()

def evaluate_and_plot_cell_counts(models_path, test_image_dir, device, patch_size=256, stride=256):
    """
    For all models in models_path, predict cell counts over all test images in test_image_dir,
    compare them to actual counts (from the corresponding XML files), and plot the comparison.
    """
    results = evaluate_cell_count_for_models(models_path, test_image_dir, device, patch_size, stride)
    plot_cell_count_comparison(results)
    return results

def call_cell_count_plot():
    test_image_dir = "data/test"
    test_mask_dir = "data/test_masks"
    patch_128_image = "data/patches_128/test_images"
    patch_128_mask = "data/patches_128/test_masks"
    patch_256_image = "data/patches_256/test_images"
    patch_256_mask = "data/patches_256/test_masks"

    transform = transforms.Compose([transforms.ToTensor()])
    print("creating dataloader")

    loader_128 = DataLoader(PatchDataset(patch_128_image, patch_128_mask, transform=transform), batch_size=4,
                            shuffle=False)
    loader_256 = DataLoader(PatchDataset(patch_256_image, patch_256_mask, transform=transform), batch_size=4,
                            shuffle=False)

    device = torch.device("cuda")
    models_path = "saved_models_128"

    r = evaluate_and_plot_cell_counts(models_path, test_image_dir, device)

    # for model_name in os.listdir(models_path):
    #     model_path = os.path.join(models_path, model_name)
    #     if not model_path.endswith(".pth"):
    #         continue
    #     try:
    #         if os.path.exists(model_path):
    #             model_data = model_name.split("_")
    #             model = UNet(in_channels=3, out_channels=2, depth=int(model_data[1]),
    #                          base_filters=int(model_data[3]))
    #
    #             model.load_state_dict(torch.load(model_path, map_location=device))
    #             print("Model " + model_path + " loaded successfully.")
    #             model.to(device)
    #
    #             for data_name, test_loader in {"patch-128": loader_128, "patch-256": loader_256}.items():
    #                 try:
    #                     print(f"Predicting {data_name}")
    #                     plot_prediction_with_cell_counts(model_name, model, test_loader, device)
    #                 except Exception as e:
    #                     print(f"Error plotting predictions: {e}")
    #     except Exception as e:
    #         print(f"Error loading model {model_name}: {e}")





def main():
    # Directories for the test patches.
    test_image_dir = "data/test"
    test_mask_dir = "data/test_masks"
    patch_128_image = "data/patches_128/test_images"
    patch_128_mask = "data/patches_128/test_masks"
    patch_256_image = "data/patches_256/test_images"
    patch_256_mask = "data/patches_256/test_masks"

    transform = transforms.Compose([transforms.ToTensor()])
    print("creating dataloader")

    # Create the test dataset and loader.
    loader_128 = DataLoader(PatchDataset(patch_128_image, patch_128_mask, transform=transform), batch_size=4,
                            shuffle=False)
    loader_256 = DataLoader(PatchDataset(patch_256_image, patch_256_mask, transform=transform), batch_size=4,
                            shuffle=True)

    device = torch.device("cpu")
    models_path = "saved_models_256"

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
                # plot_full_image_predictions_n(model_name, model, image_paths, mask_paths, device, patch_size=256,
                #                               stride=256,
                #                               n_images=n_images_to_show, save_fig=True)

                for data_name, test_loader in {"patch-128": loader_128, "patch-256": loader_256}.items():
                    try:
                        print(f"Predicting {data_name}")
                        plot_predictions(model_name, model, data_name, test_loader, device, n_images=1, save_fig=False, plt_show=True)


                    except Exception as e:
                        print(f"Error plotting predictions: {e}")
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")


if __name__ == '__main__':
    # plot_iou_dice_curves()
    # main()
    call_cell_count_plot()
