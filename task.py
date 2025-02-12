import os

import numpy as np

from train import train_model, train_model_with_params

models_to_train = [
    # {"filter": 32, "depth": 3, "min_epochs": 10, "learning_rate": 0.0001, "image_dir": "data/patches_128/train_images",
    #  "mask_dir": "data/patches_128/train_masks"},
    # {"filter": 32, "depth": 4, "min_epochs": 10, "learning_rate": 0.0001, "image_dir": "data/patches_128/train_images",
    #  "mask_dir": "data/patches_128/train_masks"},
    # {"filter": 32, "depth": 5, "min_epochs": 10, "learning_rate": 0.0001, "image_dir": "data/patches_128/train_images",
    #  "mask_dir": "data/patches_128/train_masks"},
    # {"filter": 32, "depth": 6, "min_epochs": 10, "learning_rate": 0.0001, "image_dir": "data/patches_128/train_images",
    #  "mask_dir": "data/patches_128/train_masks"},
    # {"filter": 64, "depth": 3, "min_epochs": 10, "learning_rate": 0.0001, "image_dir": "data/patches_128/train_images",
    #  "mask_dir": "data/patches_128/train_masks"},
    # {"filter": 64, "depth": 4, "min_epochs": 10, "learning_rate": 0.0001, "image_dir": "data/patches_128/train_images",
    #  "mask_dir": "data/patches_128/train_masks"},
    # {"filter": 64, "depth": 5, "min_epochs": 10, "learning_rate": 0.0001, "image_dir": "data/patches_128/train_images",
    #  "mask_dir": "data/patches_128/train_masks"},
    # {"filter": 32, "depth": 3, "min_epochs": 10, "learning_rate": 0.001, "image_dir": "data/patches_128/train_images",
    #  "mask_dir": "data/patches_128/train_masks"},
    # {"filter": 32, "depth": 4, "min_epochs": 10, "learning_rate": 0.001, "image_dir": "data/patches_128/train_images",
    #  "mask_dir": "data/patches_128/train_masks"},
    # {"filter": 32, "depth": 5, "min_epochs": 10, "learning_rate": 0.001, "image_dir": "data/patches_128/train_images",
    #  "mask_dir": "data/patches_128/train_masks"},
    # {"filter": 32, "depth": 6, "min_epochs": 10, "learning_rate": 0.001, "image_dir": "data/patches_128/train_images",
    #  "mask_dir": "data/patches_128/train_masks"},
    # {"filter": 64, "depth": 3, "min_epochs": 10, "learning_rate": 0.001, "image_dir": "data/patches_128/train_images",
    #  "mask_dir": "data/patches_128/train_masks"},
    # {"filter": 64, "depth": 4, "min_epochs": 10, "learning_rate": 0.001, "image_dir": "data/patches_128/train_images",
    #  "mask_dir": "data/patches_128/train_masks"},
    # {"filter": 64, "depth": 5, "min_epochs": 10, "learning_rate": 0.001, "image_dir": "data/patches_128/train_images",
    #  "mask_dir": "data/patches_128/train_masks"},
    # {"filter": 64, "depth": 6, "min_epochs": 10, "learning_rate": 0.001, "image_dir": "data/patches_128/train_images",
    #  "mask_dir": "data/patches_128/train_masks"},
    {"filter": 32, "depth": 3, "min_epochs": 10, "learning_rate": 0.0001, "image_dir": "data/patches_256/train_images",
     "mask_dir": "data/patches_256/train_masks"},
    {"filter": 32, "depth": 6, "min_epochs": 10, "learning_rate": 0.0001, "image_dir": "data/patches_256/train_images",
     "mask_dir": "data/patches_256/train_masks"},
    {"filter": 64, "depth": 3, "min_epochs": 10, "learning_rate": 0.0001, "image_dir": "data/patches_256/train_images",
     "mask_dir": "data/patches_256/train_masks"},
]


def main():
    for model in models_to_train:
        try:
            running_loss = train_model_with_params(model["depth"], model["filter"], model["learning_rate"],
                                                   min_epochs=model["min_epochs"], train_image_dir=model["image_dir"],
                                                   train_mask_dir=model["mask_dir"])
            print(
                f"Training completed for model with depth {model['depth']}, filter {model['filter']}, learning rate {model['learning_rate']}. Loss: {running_loss}")
        except Exception as e:
            print(f"Training failed for model with depth {model['depth']}, filter {model['filter']}, learning rate {model['learning_rate']}. Error: {e}")
            continue

    # Plot the loss curves
    loss_files = [f"saved_models/d_{model['depth']}_f_{model['filter']}_lr_{model['learning_rate']}_loss.npy" for model in models_to_train]

    plot_loss_curves(folder="saved_models")

def plot_loss_curves(folder="saved_models"):
    """
    Given a list of loss file paths (NumPy .npy files), this function loads each
    loss curve and plots them on the same figure for comparison.
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))

    loss_file_list = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith("_loss.npy")]

    for loss_file in loss_file_list:
        losses = np.load(loss_file)
        label = os.path.basename(loss_file).replace("_loss.npy", "")
        plt.plot(losses, label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # main()
    plot_loss_curves()