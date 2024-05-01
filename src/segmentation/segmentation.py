import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import torch.optim as optim
import torch
from torchmetrics.detection import MeanAveragePrecision
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms

from src.segmentation.unet_model import UNet
from src.configs.segment_config import IN_CHANNELS, OUT_CHANNELS
from src.segmentation.dice_loss import DiceLoss

from skimage.io import imread, imshow
from skimage.color import rgb2gray, label2rgb
from skimage.feature import blob_dog, blob_log, blob_doh


class SegmentationDataset(torch.utils.data.Dataset):
    in_channels = IN_CHANNELS
    out_channels = OUT_CHANNELS

    def __init__(self, imgs: list, mask: list, n_blobs: list, transforms=None):
        self.transforms = transforms
        self.imgs = imgs
        self.mask = mask
        self.n_blobs = n_blobs

    def __getitem__(self, idx):
        image = Image.open(self.imgs[idx])
        mask = Image.open(self.mask[idx]).convert("L")
        n_blobs = self.n_blobs[idx]

        if self.transforms is not None:
            image = self.transforms(image)
            mask = self.transforms(mask)

        image_tensor = transforms.ToTensor()(image)
        mask_tensor = transforms.ToTensor()(mask)

        return image_tensor, mask_tensor, n_blobs

    def __len__(self):
        return len(self.imgs)


def blob_detection(img, i):
    hotdog = 255 - img
    # hotdog = imread(img)
    print(hotdog.shape)
    hotdog = cv2.cvtColor(hotdog, cv2.COLOR_BGR2GRAY)

    blobs = blob_log(hotdog, max_sigma=30, num_sigma=10, threshold=0.5)
    # blobs = blob_dog(hotdog, max_sigma=100, threshold=0.5)
    print(len(blobs))

    fig, ax = plt.subplots()
    ax.imshow(hotdog, cmap="gray")
    for blob in blobs:
        y, x, area = blob
        ax.add_patch(plt.Circle((x, y), area * np.sqrt(2), color="r", fill=False))
    # plt.show()
    fig.savefig(f"../data/test_ima/test_mask_{i}.jpg")
    return len(blobs)


def blob_detection_1(img):
    """
    Function to detect blob in one image
    :param img:
    :return:
    """
    # Read image
    img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    img = 255 - img
    print(img.shape)

    # Set up the detector with default parameters.
    params = cv2.SimpleBlobDetector_Params()
    params.minInertiaRatio = 0.01

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs.
    keypoints = detector.detect(img)
    print(len(keypoints))

    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    blank = np.zeros((1, 1))
    blobs = cv2.drawKeypoints(img, keypoints, blank, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Show keypoints
    # fig = plt.figure(figsize=(15, 5))
    # plt.imshow(blobs)
    # fig.suptitle("title")
    cv2.imwrite(f"../key_points.jpg", blobs)
    # plt.savefig(f'../key_points.jpg')
    print("Saved file")
    return keypoints


def train_val(loader_train, loader_valid, device, num_epochs=50, learning_rate=0.001, weight_decay=0.1):
    loaders = {"train": loader_train, "valid": loader_valid}

    model = UNet(in_channels=SegmentationDataset.in_channels, out_channels=SegmentationDataset.out_channels)
    model.to(device)

    dsc_loss = DiceLoss()
    metric = MeanAveragePrecision(iou_type="segm")
    best_validation_dsc = 0.0

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # logger = logging
    loss_train = []
    loss_valid = []
    loss_df = []
    step = 0
    for epoch in tqdm(range(num_epochs), total=num_epochs):
        for phase in ["train", "valid"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            for i, data in enumerate(loaders[phase]):
                if phase == "train":
                    step += 1

                images, masks_true, n_blobs = data
                images, masks_true = images.to(device), masks_true.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    masks_pred = model(images)

                    loss = dsc_loss(masks_pred, masks_true)

                    if phase == "valid":
                        loss_valid.append(loss.item())
                        #
                        # preds = [
                        #     dict(
                        #         masks=mask_pred,
                        #         scores=torch.tensor([0.536]),
                        #         labels=torch.tensor([0]),
                        #     )
                        #     for mask_pred in masks_pred
                        # ]
                        # target = [
                        #     dict(
                        #         masks=mask_pred,
                        #         labels=torch.tensor([0]),
                        #     )
                        #     for mask_pred in masks_pred
                        # ]
                        # metric.update(preds, target)

                    if phase == "train":
                        loss_train.append(loss.item())
                        loss.backward()
                        optimizer.step()

                if phase == "train" and (step + 1) % 10 == 0:
                    # log_loss_summary(logger, loss_train, step)
                    print(loss_train[-1])
                    # loss_train = []

            if phase == "valid":
                # log_loss_summary(logger, loss_valid, step, prefix="val_")
                mean_dsc = np.mean(loss_valid)
                if mean_dsc > best_validation_dsc:
                    best_validation_dsc = mean_dsc
                    # torch.save(model.state_dict(), os.path.join(, "model.pt"))
                # loss_valid = []

        loss_df.append({"epoch": epoch, "train": np.mean(loss_train), "test": np.mean(loss_valid)})
        loss_train = []
        loss_valid = []

    loss_df = pd.DataFrame(loss_df)
    print("Best validation mean DSC: {:4f}".format(best_validation_dsc))
    torch.save(model, "../model_inference.pt")
    return model, loss_df, metric


def predict_segmentation(loader_test, device, model=None):
    if model is None:
        model = torch.load("../model_inference.pt")
    model.to(device)
    test_loss_list = []
    model.eval()
    dsc_loss = DiceLoss()
    metric = MeanAveragePrecision(iou_type="segm")
    n_blob_pred = []
    for i, data in enumerate(loader_test):
        images, masks_true, n_blobs_true = data
        images, masks_true = images.to(device), masks_true.to(device)

        masks_pred = model(images)
        loss = dsc_loss(masks_pred, masks_true)
        masks_pred = masks_pred.to(torch.device("cpu")).detach().numpy()
        for i, mask in enumerate(masks_pred):
            mask = mask.transpose(1, 2, 0)
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
            n_blob_pred.append(blob_detection(mask, i))

        # gather the stats from all processes
        test_loss_list.append(loss)

    # torch.set_num_threads(n_threads)
    return n_blob_pred, metric, test_loss_list
