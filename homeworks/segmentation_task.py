import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from src.segmentation import train_val


def main():
    IMAGE_SIZE = (150, 150)
    # prepare dataset
    print("prepare dataset")

    # prepare torch dataset
    train_small_loader, valid_loader = train_val_split(train_images, train_labels, 0.3)
    train_loader = images_to_torch_dataset(train_images, train_labels)
    test_loader = images_to_torch_dataset(test_images, test_labels)
    loader_train, loader_valid = data_loaders(args)
    loaders = {"train": loader_train, "valid": loader_valid}

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    # check parameters
    trainval_loss_list_alex, val_acc_list_alex = train_val(
        train_small_loader, valid_loader, device, num_epochs=10, learning_rate=1.0e-4, weight_decay=0.1
    )
    plot_train_val(trainval_loss_list_alex, val_acc_list_alex, title="U-NET model")

    df = pd.DataFrame.from_dict(
        {"original": train_loss_list, "alex": train_loss_list_alex, "feature_ex": train_loss_list_feat}
    )

    fig = sns.lineplot(df)
    plt.xlabel("epoch")
    plt.ylabel("train loss")
    plt.savefig("../unet_train_compare.jpg")
