import json

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
import logging
import torchvision.transforms as transforms
from homeworks.classification_task1 import plot_train_val
from src.segmentation.segmentation import train_val, SegmentationDataset, blob_detection
from src.configs.segment_config import IMAGE_SIZE, train_path, test_path, train_files, test_files


def load_images(amount_dict_file, img_dir):
    with open(amount_dict_file) as f:
        amount_dict = json.load(f)
    loaded_images = []
    loaded_masks = []
    loaded_target_amount = []
    for img_name, amount in amount_dict.items():
        img_filename = os.path.join(img_dir, img_name)
        loaded_images.append(img_filename)
        mask_filename = os.path.join(img_dir, f'mask_{img_name}')
        loaded_masks.append(mask_filename)
        loaded_target_amount.append(amount)

    return loaded_images, loaded_masks, loaded_target_amount


def data_loaders():
    transformers = transforms.Compose([transforms.Resize(IMAGE_SIZE)])
    print("Load train data")
    loaded_images_train, loaded_target_regions_train, train_target_amount = load_images(train_files, train_path)
    train_data = SegmentationDataset(loaded_images_train, loaded_target_regions_train, transformers)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=5, shuffle=True) #, collate_fn=collate_fn)


    print("Load test data")
    loaded_images_test, loaded_target_regions_test, test_target_amount = load_images(test_files, test_path)
    test_data = SegmentationDataset(loaded_images_test, loaded_target_regions_test, transformers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=5, shuffle=True) #, collate_fn=collate_fn)

    # print(test_target_amount[1])
    k=0
    for i in range(len(test_target_amount)):
        n_blobs = blob_detection(loaded_target_regions_test[i])
        if n_blobs != test_target_amount[i]:
            k+=1
    print(f"Number of mismatches in test is {k}")
    print(f"Len of test {len(test_target_amount)}")

    return train_loader, test_loader


def main():
    # prepare dataset
    print("prepare dataset")
    # prepare torch dataset
    train_loader, test_loader = data_loaders()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")
    valid_loader = test_loader

    # check parameters
    # print("Start train")
    # trainval_loss_list, val_acc_list = train_val(
    #     train_loader, valid_loader, device, num_epochs=2, learning_rate=1.0e-3, weight_decay=0.1
    # )
    # plot_train_val(trainval_loss_list, val_acc_list, title="U-NET model train-validation")


    # train_loss_list, test_acc_list = train_val(
    #     train_loader, test_loader, device, num_epochs=10, learning_rate=1.0e-4, weight_decay=0.1
    # )
    # plot_train_val(train_loss_list, test_acc_list, title="U-NET model test")


if __name__ == "__main__":
    print(torch.backends.mps.is_available())
    main()
