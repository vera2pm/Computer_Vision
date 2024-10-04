import os
import json
import cv2
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import lightning as L
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS

import torch
import torch.optim as optim
from torchmetrics.detection import MeanAveragePrecision

from src.segmentation.blob_detection import blob_detection
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.configs.segment_config import IN_CHANNELS, OUT_CHANNELS
from src.segmentation.dice_loss import DiceLoss


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


class SegmentationDataset(torch.utils.data.Dataset):
    in_channels = IN_CHANNELS
    out_channels = OUT_CHANNELS

    def __init__(self, imgs: list, mask: list, n_blobs: list, transforms=None):
        self.transforms = transforms
        self.imgs = imgs
        self.mask = mask
        self.n_blobs = n_blobs

    def __getitem__(self, idx):
        image = cv2.imread(self.imgs[idx])
        mask = cv2.imread(self.mask[idx], cv2.IMREAD_GRAYSCALE)
        n_blobs = self.n_blobs[idx]

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

        if self.transforms is not None:
            transformed = self.transforms(image=image, mask=mask)
            image_tensor = transformed["image"]
            mask_tensor = transformed["mask"]

        mask_tensor = mask_tensor.unsqueeze(0)
        mask_tensor = mask_tensor.float() / 255  # normalize image from 0-255 to 0-1 for dice loss
        image_tensor = image_tensor.float() / 255

        # image_tensor = transforms.ToTensor()(image)
        # mask_tensor = transforms.ToTensor()(mask)

        return image_tensor, mask_tensor, n_blobs

    def __len__(self):
        return len(self.imgs)


class SegmentationDataModule(L.LightningDataModule):
    def __init__(self, train_path, test_path, train_files, test_files, transformers, batch_size: int):
        super().__init__()
        self.train_path = train_path
        self.test_path = test_path
        self.train_files = train_files
        self.test_files = test_files
        self.batch_size = batch_size
        self.transformers = transformers

    def setup(self, stage: str) -> None:
        if stage == "fit":
            loaded_images_train, loaded_target_regions_train, train_target_amount = load_images(
                self.train_files, self.train_path
            )

            train_data = SegmentationDataset(
                loaded_images_train, loaded_target_regions_train, train_target_amount, self.transformers
            )
            print("Split to train and validation subsets")
            self.train_subset, self.val_subset = torch.utils.data.random_split(
                train_data, [0.8, 0.2], generator=torch.Generator().manual_seed(42)
            )
        if stage == "test":
            loaded_images_test, loaded_target_regions_test, test_target_amount = load_images(
                self.test_files, self.test_path
            )
            self.test_data = SegmentationDataset(
                loaded_images_test, loaded_target_regions_test, test_target_amount, self.transformers
            )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return torch.utils.data.DataLoader(
            self.train_subset, batch_size=self.batch_size, shuffle=True, num_workers=7, persistent_workers=True
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return torch.utils.data.DataLoader(
            self.val_subset, batch_size=self.batch_size, shuffle=False, num_workers=7, persistent_workers=True
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return torch.utils.data.DataLoader(
            self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=7, persistent_workers=True
        )


class Segmentation(L.LightningModule):
    def __init__(self, model, learning_rate, weight_decay):
        super().__init__()
        self.model = model
        self.save_hyperparameters()  # ignore=["model"])

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.dsc_loss = DiceLoss()
        metric = MeanAveragePrecision(iou_type="segm")

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer

    def _calculate_loss_per_batch(self, data):
        images, masks_true, n_blobs = data
        masks_pred = self.model(images)
        loss = self.dsc_loss(masks_pred, masks_true)
        return loss

    def training_step(self, batch):
        # training_step defines the train loop.
        loss = self._calculate_loss_per_batch(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch):
        # this is the validation loop
        val_loss = self._calculate_loss_per_batch(batch)
        self.log("val_loss", val_loss, prog_bar=True)

    def test_step(self, batch):
        images, masks_true, n_blobs = batch

        masks_pred = self.model(images)
        loss = self.dsc_loss(masks_pred, masks_true)
        masks_pred = masks_pred.to(torch.device("cpu")).detach().numpy()
        n_blob_pred = []
        for i, mask in enumerate(masks_pred):
            mask = mask.transpose(1, 2, 0)
            mask = cv2.cvtColor(np.float32(mask), cv2.COLOR_RGB2BGR)
            n_blob_pred.append(blob_detection(mask, i))
        n_blobs = n_blobs.to(torch.device("cpu")).detach().numpy()

        self.log("test_MAE", mean_absolute_error(n_blobs, n_blob_pred), prog_bar=True)

        self.log("test_loss", loss, prog_bar=True)
        tensorboard_logger = self.logger.experiment
        for i, pred in enumerate(masks_pred):
            tensorboard_logger.add_image("test_preds", pred, i)

    def predict_step(self, batch):
        images, masks_true, n_blobs = batch

        masks_pred = self.model(images)
        masks_pred = masks_pred.to(torch.device("cpu")).detach().numpy()
        n_blob_pred = []
        for i, mask in enumerate(masks_pred):
            mask = mask.transpose(1, 2, 0)
            mask = cv2.cvtColor(np.float32(mask), cv2.COLOR_RGB2BGR)  # blob detection needs BGR, when mask is in RGB
            n_blob_pred.append(blob_detection(mask, i))
        return n_blob_pred
