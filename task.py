from train import train_model, train_model_with_params


def main():
    models_to_train = [
        # {"Filter": 32, "Depth": 3, "learning_rate": 0.0001},
        # {"Filter": 32, "Depth": 4, "Epochs": 10, "learning_rate": 0.0001},
        # {"Filter": 64, "Depth": 3, "Epochs": 10, "learning_rate": 0.0001},
        # {"Filter": 64, "Depth": 4, "Epochs": 10, "learning_rate": 0.0001},
        # {"Filter": 32, "Depth": 6, "Epochs": 10, "learning_rate": 0.001},
        {"Filter": 64, "Depth": 6, "Epochs": 10, "learning_rate": 0.0001},
    ]

    for model in models_to_train:
        running_loss = train_model_with_params(model["Depth"], model["Filter"], model["learning_rate"])
        print(f"Training completed for model with depth {model['Depth']}, filter {model['Filter']}, learning rate {model['learning_rate']}. Loss: {running_loss}")


if __name__ == '__main__':
    main()