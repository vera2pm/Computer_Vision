import os
import cv2
import matplotlib.pyplot as plt
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

    def __init__(self, imgs: list, mask: list, transforms=None):
        self.transforms = transforms
        self.imgs = imgs
        self.mask = mask

    def __getitem__(self, idx):
        image = Image.open(self.imgs[idx])
        mask = Image.open(self.mask[idx])

        if self.transforms is not None:
            image = self.transforms(image)
            mask = self.transforms(mask)

        image_tensor = transforms.ToTensor()(image)
        mask_tensor = transforms.ToTensor()(mask)

        return image_tensor, mask_tensor

    def __len__(self):
        return len(self.imgs)


def blob_detection(img):
    hotdog = imread(img)
    hotdog = 255 - hotdog
    print(hotdog.shape)

    blobs = blob_log(hotdog, max_sigma=30, num_sigma=10, threshold=0.5)
    # blobs = blob_dog(hotdog, max_sigma=100, threshold=0.5)
    print(len(blobs))

    # fig, ax = plt.subplots()
    # ax.imshow(hotdog, cmap='gray')
    # for blob in blobs:
    #     y, x, area = blob
    #     ax.add_patch(plt.Circle((x, y), area * np.sqrt(2), color='r',
    #                             fill=False))
    # plt.show()
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

    # _, thresholdimage = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)
    # print(img.shape)
    # img = thresholdimage

    # Set up the detector with default parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 10
    params.maxThreshold = 200

    # Filter by Area.
    params.filterByArea = True
    params.maxArea = 1500

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.1

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.87

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.01

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs.
    keypoints = detector.detect(img)
    # features = cv2.SIFT_create(contrastThreshold=0.2, edgeThreshold=5)
    #
    # keypoints = features.detect(img, None)
    print(len(keypoints))

    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    blank = np.zeros((1, 1))
    blobs = cv2.drawKeypoints(
        img, keypoints, blank, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    # Show keypoints
    # fig = plt.figure(figsize=(15, 5))
    # plt.imshow(blobs)
    # fig.suptitle("title")
    cv2.imwrite(f'../key_points.jpg', blobs)
    # plt.savefig(f'../key_points.jpg')
    print("Saved file")
    return keypoints


def train_val(loader_train, loader_valid, device, num_epochs=50, learning_rate=0.001, weight_decay=0.1):
    loaders = {"train": loader_train, "valid": loader_valid}

    unet = UNet(in_channels=SegmentationDataset.in_channels, out_channels=SegmentationDataset.out_channels)
    unet.to(device)

    dsc_loss = DiceLoss()
    best_validation_dsc = 0.0

    optimizer = optim.Adam(unet.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # logger = logging
    loss_train = []
    loss_valid = []

    step = 0

    for epoch in tqdm(range(num_epochs), total=num_epochs):
        for phase in ["train", "valid"]:
            if phase == "train":
                unet.train()
            else:
                unet.eval()

            validation_pred = []
            validation_true = []

            for i, data in enumerate(loaders[phase]):
                if phase == "train":
                    step += 1

                images, masks_true = data
                # images = list(image.to(device) for image in images)
                # masks_true = list(mask.to(device) for mask in masks_true)
                images, masks_true = images.to(device), masks_true.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    masks_pred = unet(images)

                    loss = dsc_loss(masks_pred, masks_true)

                    if phase == "valid":
                        loss_valid.append(loss.item())
                        # y_pred_np = y_pred.detach().cpu().numpy()
                        # validation_pred.extend([y_pred_np[s] for s in range(y_pred_np.shape[0])])
                        # y_true_np = y_true.detach().cpu().numpy()
                        # validation_true.extend([y_true_np[s] for s in range(y_true_np.shape[0])])
                        # if (epoch % args.vis_freq == 0) or (epoch == num_epochs - 1):
                        # if i * args.batch_size < args.vis_images:
                        #     tag = "image/{}".format(i)
                        #     num_images = args.vis_images - i * args.batch_size
                        #     logger.image_list_summary(
                        #         tag,
                        #         log_images(x, y_true, y_pred)[:num_images],
                        #         step,
                        #     )

                    if phase == "train":
                        loss_train.append(loss.item())
                        loss.backward()
                        optimizer.step()

                if phase == "train" and (step + 1) % 10 == 0:
                    # log_loss_summary(logger, loss_train, step)
                    print(loss_train[-1])
                    loss_train = []

            if phase == "valid":
                # log_loss_summary(logger, loss_valid, step, prefix="val_")
                mean_dsc = np.mean(loss_valid)
                # logger.scalar_summary("val_dsc", mean_dsc, step)
                if mean_dsc > best_validation_dsc:
                    best_validation_dsc = mean_dsc
                    # torch.save(unet.state_dict(), os.path.join(, "unet.pt"))
                loss_valid = []

    print("Best validation mean DSC: {:4f}".format(best_validation_dsc))
    return loss_train, loss_valid


def eval_segmentation(test_data, model, device):
    model.to(device)
    test_loss_list = []
    model.eval()
    metric = MeanAveragePrecision(iou_type="bbox")
    for i, (images, targets) in enumerate(test_data):
        images = list(img.to(device) for img in images)

        predictions = model(images)
        predictions = [{k: v.to(device) for k, v in t.items()} for t in predictions]

        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        print(targets)

        # gather the stats from all processes
        test_loss_list.append(metric(predictions, targets))

    # torch.set_num_threads(n_threads)
    return predictions, metric, test_loss_list
