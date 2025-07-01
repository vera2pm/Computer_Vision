import random
from typing import List, Optional
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import random_split, DataLoader
import torch.optim as optim
import torchvision.models.detection.backbone_utils
from torchvision import transforms
from PIL import Image
from sklearn.metrics import accuracy_score

from torchvision.models import ResNet18_Weights
from tqdm import tqdm
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
import sys

sys.path.append("../")
from src.metric_learning.sampler import PKSampler, batch_hard_triplet_loss
from src.utils import get_device

data_path = "../data/Stanford_Online_Products/"


def load_image(image_path):
    image = Image.open(data_path + image_path)
    return image


class TripletMarginLoss(nn.Module):
    def __init__(self, margin=1.0, p=2.0):
        super().__init__()
        self.margin = margin
        self.p = p

        self.loss_fn = batch_hard_triplet_loss

    def forward(self, embeddings, labels):
        return self.loss_fn(labels, embeddings, self.margin, self.p)


class TripletDataset(torch.utils.data.Dataset):
    def __init__(
        self, data_path: str, is_train: bool = True, transform: Optional[List] = None, is_super_label: bool = False
    ):
        self.data_table = pd.read_csv(data_path)
        self.imgs = self.data_table["path"]
        self.label = self.data_table["class_id"]
        self.super_label = self.data_table["super_class_id"]
        self.is_train = is_train
        self.transform = transform
        self.is_super_label = is_super_label
        self.target = self.super_label if self.is_super_label else self.label

    def __getitem__(self, idx):
        image = load_image(self.imgs[idx])
        label = self.label[idx]
        super_label = self.super_label[idx]
        image_tensor = self.transform(image)
        out_label = super_label if self.is_super_label else label

        if self.is_train:
            if self.is_super_label:
                positive_list = self.imgs[(self.data_table.index != idx) & (self.super_label == super_label)].tolist()
                negative_list = self.imgs[(self.super_label != super_label)].tolist()
            else:
                positive_list = self.imgs[(self.data_table.index != idx) & (self.label == label)].tolist()
                negative_list = self.imgs[(self.super_label == super_label) & (self.label != label)].tolist()

            pos_path = random.choice(positive_list)
            positive_image = load_image(pos_path)
            neg_path = random.choice(negative_list)
            negative_image = load_image(neg_path)

            positive_tensor = self.transform(positive_image)
            negative_tensor = self.transform(negative_image)

            return image_tensor, out_label, positive_tensor, negative_tensor

        else:
            return image_tensor, out_label

    def __len__(self):
        return len(self.imgs)


class TripletDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_path,
        valid_path,
        test_path,
        transformers,
        labels_per_batch: int,
        samples_per_label: int,
        is_super_label: bool,
        online_sample: bool,
    ):
        super().__init__()
        self.train_path = train_path
        self.valid_path = valid_path
        self.test_path = test_path
        self.p = labels_per_batch
        self.k = samples_per_label
        self.batch_size = self.p * self.k
        self.transformers = transformers
        self.is_super_label = is_super_label
        self.online_sample = online_sample

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_data_ds = TripletDataset(
                self.train_path, is_train=False, transform=self.transformers, is_super_label=self.is_super_label
            )

            self.valid_data_ds = TripletDataset(
                self.valid_path, is_train=False, transform=self.transformers, is_super_label=self.is_super_label
            )

        if stage == "predict":
            self.test_data_ds = TripletDataset(
                self.test_path, is_train=False, transform=self.transformers, is_super_label=self.is_super_label
            )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        sampler = PKSampler(self.train_data_ds.target, self.p, self.k) if self.online_sample else None
        return torch.utils.data.DataLoader(
            self.train_data_ds,
            batch_size=self.batch_size,
            # shuffle=True,
            num_workers=7,
            persistent_workers=True,
            sampler=sampler,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        sampler = PKSampler(self.valid_data_ds.target, self.p, self.k) if self.online_sample else None
        return torch.utils.data.DataLoader(
            self.valid_data_ds,
            batch_size=self.batch_size,
            # shuffle=False,
            num_workers=7,
            persistent_workers=True,
            sampler=sampler,
        )

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return torch.utils.data.DataLoader(
            self.test_data_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=7,
            persistent_workers=True,
        )


class LitMetricLearning(L.LightningModule):
    def __init__(self, learning_rate, weight_decay, online_sample=True):
        super().__init__()

        self.model = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        num_filters = self.model.fc.in_features
        self.model.fc = nn.Sequential(nn.Linear(num_filters, 512), nn.ReLU(), nn.Linear(512, 128))
        self.save_hyperparameters()  # ignore=["model"])
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.online_sample = online_sample
        if self.online_sample:
            self.triplet_loss_online = TripletMarginLoss(margin=1.0)
            self.distance_function = torch.cdist
        else:
            self.triplet_loss = nn.TripletMarginWithDistanceLoss(margin=1.0, swap=True)
            self.distance_function = self.triplet_loss.distance_function

    def forward(self, x):
        features = self.model(x)
        features = F.normalize(features, p=2, dim=1)
        return features

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer

    def _calculate_loss_per_batch(self, data):
        if self.online_sample:
            anchor_img, anchor_label = data
            anchor_img_emb = self.model(anchor_img)
            loss = self.triplet_loss_online(anchor_img_emb, anchor_label)
        else:
            anchor_img, anchor_label, positive_img, negative_img = data

            anchor_img_emb = self.model(anchor_img)
            positive_img_emb = self.model(positive_img)
            negative_img_emb = self.model(negative_img)

            loss = self.triplet_loss(anchor_img_emb, positive_img_emb, negative_img_emb)
        return loss

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        loss = self._calculate_loss_per_batch(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        val_loss = self._calculate_loss_per_batch(batch)
        self.log("val_loss", val_loss, prog_bar=True)

    def predict_step(self, batch, batch_idx):
        test_images, test_labels = batch
        test_embs = self.model(test_images)
        return test_embs, test_labels


def find_similar_items(predictions_labels, distance_function, logger=None):
    true_labels = []
    predictions = []
    images = []
    for preds, test_labels in tqdm(predictions_labels):
        predictions.append(preds)
        true_labels.append(test_labels)

    predictions = torch.cat(predictions, dim=0).float()
    true_labels = torch.cat(true_labels, dim=0).to(torch.device("cpu")).detach().numpy()
    # tensorboard_logger = logger

    pred_labels = []
    for i, pred in tqdm(enumerate(predictions)):
        distances = distance_function(pred.unsqueeze(0), predictions).to(torch.device("cpu")).detach().numpy()
        pred_im_idx = np.argsort(distances[0])[1]
        pred_labels.append(np.take(true_labels, pred_im_idx))
        # if i%10 == 0:
        #     tensorboard_logger.add_image("anchor_preds", images[i], i)
        #     tensorboard_logger.add_image("predict_preds", images[pred_im_idx], i)

    print(accuracy_score(true_labels, pred_labels))
    print(true_labels[:10])
    print(pred_labels[:10])


def plot_loss(train_loss_list, valid_loss_list, model_name):
    df = pd.DataFrame(columns=["train", "valid"])
    df["train"] = train_loss_list
    df["valid"] = valid_loss_list
    sns.lineplot(data=df)
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.savefig(f"../{model_name}.jpg")


def main():
    dev = get_device()

    # get data
    train_path = f"{data_path}Ebay_train_train_preproc.csv"
    valid_path = f"{data_path}Ebay_train__val_preproc.csv"
    test_path = f"{data_path}Ebay_test_preproc.csv"
    transformers = transforms.Compose(
        [
            transforms.Resize((512, 512)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ]
    )
    data_module = TripletDataModule(
        train_path, valid_path, test_path, transformers, 4, 10, is_super_label=True, online_sample=True
    )

    model_name = "Similarity"
    checkpoint_callback = ModelCheckpoint(dirpath=f"../logs/{model_name}", save_top_k=1, monitor="val_loss")
    logger = TensorBoardLogger("../logs/", name=model_name)
    trainer = L.Trainer(
        accelerator=dev, devices=1, max_epochs=150, logger=logger, callbacks=[checkpoint_callback], precision="16-mixed"
    )

    learning = LitMetricLearning(learning_rate=1e-7, weight_decay=0.1, online_sample=True)
    trainer.fit(model=learning, datamodule=data_module)

    learning = LitMetricLearning.load_from_checkpoint(checkpoint_callback.best_model_path)
    predictions = trainer.predict(learning, datamodule=data_module)
    find_similar_items(predictions, learning.distance_function)


def predict_clip(dataset, model, dev):
    predictions_labels = []

    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size=20)):
            features = model.encode_image(images.to(dev))

            predictions_labels.append(tuple([features, labels]))

    return predictions_labels


def main_clip():
    dev = get_device()
    model, preprocess = clip.load("ViT-B/32", dev)

    # get data
    # train_path = f"{data_path}Ebay_train_train_preproc.csv"
    # valid_path = f"{data_path}Ebay_train__val_preproc.csv"
    test_path = f"{data_path}Ebay_test_preproc.csv"
    test_data_ds = TripletDataset(test_path, is_train=False, transform=preprocess, is_super_label=True)

    # get embeddings and true labels
    predictions_labels = predict_clip(test_data_ds, model, dev)

    find_similar_items(predictions_labels, torch.cdist)


if __name__ == "__main__":
    # main()
    main_clip()
