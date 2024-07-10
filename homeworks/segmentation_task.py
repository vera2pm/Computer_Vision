import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
import torchvision.transforms as transforms
from src.segmentation.segmentation import train_val, SegmentationDataset, blob_detection, predict_segmentation
from src.configs.segment_config import IMAGE_SIZE, train_path, test_path, train_files, test_files


def plot_train_val(df, title="Segmentation UNET model"):
    fig = plt.figure(figsize=(15, 5))
    ax0 = fig.add_subplot(121, title="Train Loss", xlabel="Epoch")
    # ax1 = fig.add_subplot(122, title="Validation Loss", xlabel="Epoch")
    sns.lineplot(data=df.set_index("epoch"), ax=ax0)
    fig.suptitle(title)
    plt.savefig(f"../{title}.jpg")


def load_images(amount_dict_file, img_dir):
    with open(amount_dict_file) as f:
        amount_dict = json.load(f)
    loaded_images = []
    loaded_masks = []
    loaded_target_amount = []
    for img_name, amount in amount_dict.items():
        img_filename = os.path.join(img_dir, img_name)
        loaded_images.append(img_filename)
        mask_filename = os.path.join(img_dir, f"mask_{img_name}")
        loaded_masks.append(mask_filename)
        loaded_target_amount.append(amount)

    return loaded_images, loaded_masks, loaded_target_amount


def data_loaders():
    transformers = transforms.Compose([transforms.Resize(IMAGE_SIZE)])
    print("Load train data")
    loaded_images_train, loaded_target_regions_train, train_target_amount = load_images(train_files, train_path)
    train_data = SegmentationDataset(
        loaded_images_train, loaded_target_regions_train, train_target_amount, transformers
    )
    print("Split to train and validation subsets")
    train_subset, val_subset = torch.utils.data.random_split(
        train_data, [0.8, 0.2], generator=torch.Generator().manual_seed(42)
    )
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=20, shuffle=True)  # , collate_fn=collate_fn)
    valid_loader = torch.utils.data.DataLoader(val_subset, batch_size=20, shuffle=True)
    train_full_loader = torch.utils.data.DataLoader(train_data, batch_size=30, shuffle=True)

    print("Load test data")
    loaded_images_test, loaded_target_regions_test, test_target_amount = load_images(test_files, test_path)
    test_data = SegmentationDataset(loaded_images_test, loaded_target_regions_test, test_target_amount, transformers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=5, shuffle=False)  # , collate_fn=collate_fn)

    # k = 0
    # for i in range(10):
    #     n_blobs = blob_detection(loaded_target_regions_test[i], i)
    #     if n_blobs != test_target_amount[i]:
    #         # print(test_target_amount[i] - n_blobs)
    #         k += 1
    # print(f"Number of mismatches in test is {k}")
    # print(f"Len of test {len(test_target_amount)}")

    return train_full_loader, train_loader, valid_loader, test_loader


def main():
    # prepare dataset
    print("prepare dataset")
    # prepare torch dataset
    train_full_loader, train_loader, valid_loader, test_loader = data_loaders()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")
    # valid_loader = test_loader

    # check parameters
    print("Start train")
    model = None
    model_path = None
    model, loss_df, metric = train_val(
        train_loader, valid_loader, device, num_epochs=10, learning_rate=1.0e-4, weight_decay=0.1
    )
    print(loss_df)
    plot_train_val(loss_df, title="U-NET model train-validation")

    model_path = "../model_inference.pt"
    model, loss_df, metric = train_val(
        train_full_loader,
        test_loader,
        device,
        num_epochs=5,
        learning_rate=1.0e-4,
        weight_decay=0.1,
        model_path=model_path,
    )
    print(loss_df)
    plot_train_val(loss_df, title="U-NET model train-test")

    print("Test:")
    n_blob_pred, metric, test_loss_list, blobs_error = predict_segmentation(test_loader, device, model)
    print(test_loss_list)
    # print(f"RMSE blobs: {blobs_error}")
    # print(metric.compute())
    # train_loss_list, test_acc_list = train_val(
    #     train_loader, test_loader, device, num_epochs=10, learning_rate=1.0e-4, weight_decay=0.1
    # )
    # plot_train_val(train_loss_list, test_acc_list, title="U-NET model test")


if __name__ == "__main__":
    print(torch.backends.mps.is_available())
    main()
