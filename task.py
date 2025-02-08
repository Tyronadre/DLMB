import os

import numpy as np

from train import train_model, train_model_with_params


def main():
    models_to_train = [
        {"filter": 32, "depth": 6, "learning_rate": 0.0001, "min_epochs": 10},
        # {"filter": 32, "depth": 4, "min_epochs": 10, "learning_rate": 0.0001},
        # {"filter": 64, "depth": 3, "min_epochs": 10, "learning_rate": 0.0001},
        # {"filter": 64, "depth": 4, "min_epochs": 10, "learning_rate": 0.0001},
        # {"filter": 32, "depth": 6, "min_epochs": 10, "learning_rate": 0.001},
        # {"filter": 64, "depth": 6, "min_epochs": 10, "learning_rate": 0.0001},
    ]

    for model in models_to_train:
        running_loss = train_model_with_params(model["depth"], model["filter"], model["learning_rate"],
                                               min_epochs=model["min_epochs"], train_image_dir="data/patches_128/train_images",
                                               train_mask_dir="data/patches_128/train_masks")
        print(
            f"Training completed for model with depth {model['Depth']}, filter {model['Filter']}, learning rate {model['learning_rate']}. Loss: {running_loss}")


def plot_loss_curves(loss_file_list):
    """
    Given a list of loss file paths (NumPy .npy files), this function loads each
    loss curve and plots them on the same figure for comparison.
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
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
    main()
