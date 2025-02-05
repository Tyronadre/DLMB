import glob
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
from skimage.util import view_as_windows
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm

########################################
# Patch Extraction Utility
########################################

def extract_patches(image_array, patch_size, stride):
    """
    Extract overlapping patches from an image array.
    :param image_array: numpy array of shape (H, W, C) or (H, W)
    :param patch_size: tuple (patch_height, patch_width)
    :param stride: tuple (stride_y, stride_x)
    :return: numpy array of patches with shape (num_patches, patch_height, patch_width, C) or similar.
    """
    # For grayscale images, add a channel dimension for consistency.
    if image_array.ndim == 2:
        image_array = image_array[..., np.newaxis]
    # Use skimage's view_as_windows to get sliding windows (patches)
    patches = view_as_windows(image_array, patch_size + (image_array.shape[-1],), step=stride + (1,))
    # patches.shape = (n_y, n_x, 1, patch_height, patch_width, C); squeeze the extra dim.
    patches = patches.reshape(-1, patch_size[0], patch_size[1], image_array.shape[-1])
    return patches

def save_patches(image_dir, mask_dir, out_image_dir, out_mask_dir, patch_size=(256, 256), stride=(256, 256)):
    """
    Loop over images and masks in the given directories, extract patches, and save them to disk.
    """
    os.makedirs(out_image_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)

    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.tif")))
    mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.tif")))

    patch_id = 0
    for img_path, mask_path in zip(image_paths, mask_paths):
        # Open images using PIL and convert to numpy arrays.
        img = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))  # grayscale

        # Extract patches
        img_patches = extract_patches(img, patch_size, stride)
        mask_patches = extract_patches(mask, patch_size, stride)

        assert len(img_patches) == len(mask_patches), "Number of image and mask patches must match."

        for i in range(len(img_patches)):
            # Save image patch and mask patch as .npy files.
            np.save(os.path.join(out_image_dir, f"patch_{patch_id}.npy"), img_patches[i])
            np.save(os.path.join(out_mask_dir, f"patch_{patch_id}.npy"), mask_patches[i])
            patch_id += 1
    print(f"Saved {patch_id} patches.")

########################################
# Data Augmentation: Pair Transforms
########################################

class ComposePair:
    """
    Composes several transforms that work on (image, mask) pairs.
    """

    def __init__(self, transforms_list):
        self.transforms = transforms_list

    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask

class RandomHorizontalFlipPair:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            return F.hflip(image), F.hflip(mask)
        return image, mask

class RandomVerticalFlipPair:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            return F.vflip(image), F.vflip(mask)
        return image, mask

class RandomRotationPair:
    def __init__(self, degrees=30):
        self.degrees = degrees

    def __call__(self, image, mask):
        angle = random.uniform(-self.degrees, self.degrees)
        # Use bilinear for image and nearest for mask to preserve labels.
        return F.rotate(image, angle), F.rotate(mask, angle)

class ColorJitterPair:
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.jitter = transforms.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, image, mask):
        # Only apply color jitter to image; leave mask intact.
        return self.jitter(image), mask

def get_augmentation_transforms(config):
    """
    Build a ComposePair transform from a configuration dictionary.

    config: dictionary with boolean or parameter entries. For example:
      {
          'hflip': 0.5,            # probability for horizontal flip (None or 0 disables)
          'vflip': 0.5,            # probability for vertical flip
          'rotation': 30,          # max degrees of rotation (None or 0 disables)
          'color_jitter': {        # if present, parameters for ColorJitter
              'brightness': 0.2,
              'contrast': 0.2,
              'saturation': 0.2,
              'hue': 0.1
          }
      }
    """
    transforms_list = []

    if config.get('hflip', 0) > 0:
        transforms_list.append(RandomHorizontalFlipPair(p=config['hflip']))
    if config.get('vflip', 0) > 0:
        transforms_list.append(RandomVerticalFlipPair(p=config['vflip']))
    if config.get('rotation', 0) > 0:
        transforms_list.append(RandomRotationPair(degrees=config['rotation']))
    if config.get('color_jitter', None):
        jitter_params = config['color_jitter']
        transforms_list.append(ColorJitterPair(**jitter_params))

    if transforms_list:
        return ComposePair(transforms_list)
    else:
        # If no augmentation is selected, return identity.
        return lambda image, mask: (image, mask)

########################################
# Parameterizable UNet Implementation
########################################

class DoubleConv(nn.Module):
    """Two convolution layers each followed by BatchNorm and ReLU."""

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
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
    """
    Parameterizable U-Net.

    Parameters:
      - in_channels: number of input channels (e.g. 3 for RGB).
      - out_channels: number of output channels (number of classes).
      - depth: number of downsamplings/upsamplings.
      - base_filters: number of filters for the first layer.
    """

    def __init__(self, in_channels=3, out_channels=2, depth=4, base_filters=64):
        super(UNet, self).__init__()
        self.depth = depth

        # Build encoder (contracting path)
        self.encoders = nn.ModuleList()
        filters = base_filters
        for i in range(depth):
            if i == 0:
                self.encoders.append(DoubleConv(in_channels, filters))
            else:
                self.encoders.append(DoubleConv(filters // 2, filters))
            filters *= 2

        # Bottleneck layer
        self.bottleneck = DoubleConv(filters // 2, filters)

        # Build decoder (expanding path)
        self.decoders = nn.ModuleList()
        for i in range(depth):
            self.decoders.append(nn.ConvTranspose2d(filters, filters // 2, kernel_size=2, stride=2))
            self.decoders.append(DoubleConv(filters, filters // 2))
            filters //= 2

        self.final_conv = nn.Conv2d(filters, out_channels, kernel_size=1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        enc_features = []

        # Encoder path
        for encoder in self.encoders:
            x = encoder(x)
            enc_features.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path
        for i in range(self.depth):
            x = self.decoders[2 * i](x)
            # Get corresponding feature map from encoder (reverse order)
            enc_feat = enc_features[-(i + 1)]
            # If necessary, pad x to match dimensions
            if x.size() != enc_feat.size():
                diffY = enc_feat.size()[2] - x.size()[2]
                diffX = enc_feat.size()[3] - x.size()[3]
                x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                              diffY // 2, diffY - diffY // 2])
            x = torch.cat([enc_feat, x], dim=1)
            x = self.decoders[2 * i + 1](x)

        return self.final_conv(x)

########################################
# Integrating Augmentations into the Dataset
########################################

class PatchDataset(Dataset):
    def __init__(self, image_patch_dir, mask_patch_dir, transform=None):
        """
        :param image_patch_dir: Directory containing npy files for image patches.
        :param mask_patch_dir: Directory containing npy files for mask patches.
        :param transform: Optional transform to apply to the images (and masks if desired)
        """
        self.image_files = sorted(glob.glob(os.path.join(image_patch_dir, "*.npy")))
        self.mask_files = sorted(glob.glob(os.path.join(mask_patch_dir, "*.npy")))
        self.transform = transform

        assert len(self.image_files) == len(self.mask_files), "Mismatch between image and mask patches"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img = np.load(self.image_files[idx])  # shape (H, W, 3)
        mask = np.load(self.mask_files[idx])  # shape (H, W, 1)

        # Convert numpy arrays to PIL images if transforms require that.
        img = Image.fromarray(img.astype(np.uint8))
        mask = Image.fromarray(mask.squeeze().astype(np.uint8))  # squeeze out the channel dim for mask

        if self.transform:
            img = self.transform(img)
            # For the mask, you might need a separate transform. Here we convert to tensor without normalization.
            mask = transforms.ToTensor()(mask)
        else:
            img = transforms.ToTensor()(img)
            mask = transforms.ToTensor()(mask)

        # For segmentation, often masks are expected to be LongTensor with class indices.
        mask = mask.long().squeeze(0)
        return img, mask

########################################
# Training Loop
########################################

def plot_epoch_predictions(model, sample_loader, device, epoch, save_fig=False):
    """
    Run the model on a few sample patches and plot:
      - The original image
      - The ground truth mask
      - The predicted mask
    """
    model.eval()  # switch to evaluation mode
    with torch.no_grad():
        for imgs, masks in sample_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)

            # Plot first sample in the batch
            img = imgs[0].cpu().permute(1, 2, 0).numpy()
            true_mask = masks[0].cpu().numpy()
            pred_mask = preds[0].cpu().numpy()

            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            axs[0].imshow(img)
            axs[0].set_title("Original Image")
            axs[0].axis("off")

            axs[1].imshow(true_mask, cmap="gray")
            axs[1].set_title("Ground Truth Mask")
            axs[1].axis("off")

            axs[2].imshow(pred_mask, cmap="gray")
            axs[2].set_title("Predicted Mask")
            axs[2].axis("off")

            plt.suptitle(f"Epoch {epoch + 1}")
            plt.tight_layout()
            if save_fig:
                os.makedirs("epoch_predictions", exist_ok=True)
                plt.savefig(f"epoch_predictions/epoch_{epoch + 1}.png")
            plt.show()
            break  # we only plot one batch (the first) for brevity
    model.train()  # switch back to training mode

def train_model_with_epoch_predictions(model, train_loader, sample_loader, criterion, optimizer, device, num_epochs=25):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            imgs, masks = imgs.to(device), masks.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, masks)
            running_loss += loss.item() * imgs.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader.dataset):.4f}")

        plot_epoch_predictions(model, sample_loader, device, epoch, save_fig=True)
    return model

def train_model(model, train_loader, _, criterion, optimizer, device, num_epochs=25):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            imgs, masks = imgs.to(device), masks.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, masks)
            running_loss += loss.item() * imgs.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader.dataset):.4f}")
    return model

########################################
# 5. Main Execution
########################################


# Assuming PatchDataset and UNet are defined as in the previous examples.

def main():
    aug_config = {
        # 'hflip': 0.5,
        # 'vflip': 0.5,
        # 'rotation': 30,
        # 'color_jitter': {
        #     'brightness': 0.2,
        #     'contrast': 0.2,
        #     'saturation': 0.2,
        #     'hue': 0.1
        # }
    }
    train_transform = get_augmentation_transforms(aug_config)
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_image_dir = "data/patches/train_images"
    train_mask_dir = "data/patches/train_masks"
    train_dataset = PatchDataset(train_image_dir, train_mask_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    sample_indices = list(range(8))
    sample_subset = torch.utils.data.Subset(train_dataset, sample_indices)
    sample_loader = DataLoader(sample_subset, batch_size=4, shuffle=False)

    # Initialize the U-Net model.
    model = UNet(in_channels=3, out_channels=2, depth=4, base_filters=64)

    # Loss function and optimizer.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print(f"Using device: {device}")

    # Train the model, plotting predictions at each epoch.
    num_epochs = 5
    model = train_model_with_epoch_predictions(model, train_loader, sample_loader, criterion, optimizer, device,
                                               num_epochs)

    # Save the final model.
    model_name = "2.pth"
    os.makedirs("saved_models", exist_ok=True)
    torch.save(model.state_dict(), "saved_models/" + model_name)
    print("Model saved to saved_models/" + model_name)


if __name__ == '__main__':
    main()
