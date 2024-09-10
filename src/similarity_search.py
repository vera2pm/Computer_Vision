import random
import os
from typing import List, Optional
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import random_split
import torch.optim as optim
import torchvision.models.detection.backbone_utils
from torchvision import transforms
from PIL import Image
from sklearn.metrics import accuracy_score
from torch.utils.tensorboard import SummaryWriter

from torchvision.models import ResNet18_Weights
from tqdm import tqdm
from sklearn.decomposition import PCA
import lightning as L

data_path = "../data/Stanford_Online_Products/"


def load_image(image_path):
    image = Image.open(data_path + image_path)
    return image


def print_embeddings(all_embeddings, all_labels):
    # Reduce to 2D using PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(all_embeddings)

    # Plot the 2D embeddings
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=all_labels, cmap="tab10")
    plt.colorbar(scatter)
    plt.title("PCA of Embeddings")
    plt.savefig(f"../pca_embeddings.jpg")


class TripletDataset(torch.utils.data.Dataset):
    def __init__(
        self, data_path: str, is_train: bool = True, transform: Optional[List] = None, is_super_label: bool = False
    ):
        self.data_table = pd.read_csv(data_path)
        self.imgs = self.data_table["path"]
        self.label = self.data_table["class_id"]
        self.super_label = self.data_table["super_class_id"]
        self.is_train = is_train
        self.transform = transforms.Compose(transform)
        self.is_super_label = is_super_label

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


class ResnetTriplet(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        num_filters = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(nn.Linear(num_filters, 512), nn.ReLU(), nn.Linear(512, 128))

    def forward(self, x):
        features = self.resnet(x)
        features = F.normalize(features, p=2, dim=1)
        return features


class MetricLearning:
    def __init__(self, learning_rate, weight_decay, model_name):
        self.model_path = f"../{model_name}.pt"
        if os.path.exists(self.model_path):
            print("Load existing model")
            self.model = torch.load(self.model_path)
        else:
            print("Train new model")
            self.model = ResnetTriplet()

        if torch.backends.mps.is_available():
            dev = "mps"
        elif torch.cuda.is_available():
            dev = "cuda"
        else:
            dev = "cpu"
        self.device = torch.device(dev)
        print(f"device: {self.device}")

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.triplet_loss = nn.TripletMarginWithDistanceLoss(margin=1.0, swap=True)

    def calculate_loss(self, data_loader, phase):
        predictions = []
        true_labels = []
        train_loss = 0
        for i, data in tqdm(enumerate(data_loader), desc=phase):
            # Extracting images and target labels for the batch being iterated
            anchor_img, anchor_label, positive_img, negative_img = data
            anchor_img, anchor_label, positive_img, negative_img = (
                anchor_img.to(self.device),
                anchor_label.to(self.device),
                positive_img.to(self.device),
                negative_img.to(self.device),
            )
            self.optimizer.zero_grad()

            anchor_img_emb = self.model(anchor_img)
            positive_img_emb = self.model(positive_img)
            negative_img_emb = self.model(negative_img)

            loss = self.triplet_loss(anchor_img_emb, positive_img_emb, negative_img_emb)
            train_loss += loss.item()

            if phase == "train":
                loss.backward()
                self.optimizer.step()
            true_labels.extend(anchor_label.to(torch.device("cpu")).detach().numpy())
            predictions.append(anchor_img_emb)

        predictions = torch.cat(predictions, dim=0).to(torch.device("cpu")).detach().numpy()
        return train_loss, predictions, true_labels

    def train_test(self, train_loader, valid_loader, num_epochs=50):
        self.model.to(self.device)
        train_loss_list = []
        valid_loss_list = []
        for epoch in range(num_epochs):
            print(f"Epoch {epoch}")
            for phase in ["train", "valid"]:
                if phase == "train":
                    self.model.train()
                    train_loss, predictions_train, true_labels_train = self.calculate_loss(train_loader, "train")
                    train_loss_list.append(train_loss / len(train_loader))
                    print(f"Training loss = {train_loss_list[-1]}")
                else:
                    with torch.no_grad():
                        self.model.eval()
                        # Iterating over the training dataset in batches
                        valid_loss, predictions, true_labels = self.calculate_loss(valid_loader, "valid")

                        valid_loss_list.append(valid_loss / len(valid_loader))
                        print(f"Valid loss = {valid_loss_list[-1]}")

            torch.save(self.model, self.model_path)

        return train_loss_list, valid_loss_list

    def predict(self, test_loader):
        pred_labels = []
        predictions = []
        true_labels = []

        with torch.no_grad():
            model = self.model.to(self.device)
            model.eval()
            for i, data in enumerate(test_loader):
                test_images, test_labels = data
                test_images, test_labels = test_images.to(self.device), test_labels.to(self.device)

                test_embs = model(test_images)

                true_labels.extend(test_labels.to(torch.device("cpu")).detach().numpy())
                predictions.append(test_embs)

            predictions = torch.cat(predictions, dim=0)
            for i, pred in enumerate(predictions):
                distances = (
                    self.triplet_loss.distance_function(pred, predictions).to(torch.device("cpu")).detach().numpy()
                )
                pred_im_idx = np.argsort(distances)[1]  #:6]

                pred_labels.append(np.take(true_labels, pred_im_idx))

        return predictions, pred_labels, true_labels


class LitMetricLearning(L.LightningModule):
    def __init__(self, model_name, learning_rate, weight_decay):
        super().__init__()
        self.model_path = f"../{model_name}.pt"
        if os.path.exists(self.model_path):
            print("Load existing model")
            self.model = torch.load(self.model_path)
        else:
            print("Train new model")
            self.model = ResnetTriplet()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.triplet_loss = nn.TripletMarginWithDistanceLoss(margin=1.0, swap=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer

    def _calculate_loss_per_batch(self, data):
        anchor_img, anchor_label, positive_img, negative_img = data
        anchor_img, anchor_label, positive_img, negative_img = (
            anchor_img.to(self.device),
            anchor_label.to(self.device),
            positive_img.to(self.device),
            negative_img.to(self.device),
        )

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


def predict(checkpoint, test_loader, device):
    model = LitMetricLearning.load_from_checkpoint(checkpoint)
    pred_labels = []
    predictions = []
    true_labels = []

    with torch.no_grad():
        # model = model.to(device)
        model.eval()
        for i, data in enumerate(test_loader):
            test_images, test_labels = data
            # test_images, test_labels = test_images.to(device), test_labels.to(device)

            test_embs = model(test_images)

            true_labels.extend(test_labels.to(torch.device("cpu")).detach().numpy())
            predictions.append(test_embs)

        predictions = torch.cat(predictions, dim=0)
        for i, pred in enumerate(predictions):
            distances = model.triplet_loss.distance_function(pred, predictions).to(torch.device("cpu")).detach().numpy()
            pred_im_idx = np.argsort(distances)[1]

            pred_labels.append(np.take(true_labels, pred_im_idx))

    return predictions, pred_labels, true_labels


def load_data():
    train_data = f"{data_path}Ebay_train_train_preproc.csv"
    valid_data = f"{data_path}Ebay_train__val_preproc.csv"
    test_data = f"{data_path}Ebay_test_preproc.csv"
    transformers = [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
    ]

    train_data_ds = TripletDataset(train_data, is_train=True, transform=transformers, is_super_label=True)
    train_loader = torch.utils.data.DataLoader(train_data_ds, batch_size=56, shuffle=True)

    valid_data_ds = TripletDataset(valid_data, is_train=True, transform=transformers, is_super_label=True)
    valid_loader = torch.utils.data.DataLoader(valid_data_ds, batch_size=56, shuffle=True)

    test_data_ds = TripletDataset(test_data, is_train=False, transform=transformers, is_super_label=True)
    test_loader = torch.utils.data.DataLoader(test_data_ds, batch_size=40)

    return train_loader, valid_loader, test_loader


def plot_loss(train_loss_list, valid_loss_list, model_name):
    df = pd.DataFrame(columns=["train", "valid"])
    df["train"] = train_loss_list
    df["valid"] = valid_loss_list
    sns.lineplot(data=df)
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.savefig(f"../{model_name}.jpg")


def main():
    if torch.backends.mps.is_available():
        dev = "mps"
    elif torch.cuda.is_available():
        dev = "cuda"
    else:
        dev = "cpu"

    model_name = "model18_inference_1e5_superlabel"

    # get data
    train_loader, valid_loader, test_loader = load_data()

    # learning = MetricLearning(learning_rate=1e-5, weight_decay=0.1, model_name=model_name)
    # train_loss_list, valid_loss_list = learning.train_test(train_loader, valid_loader, num_epochs=50)
    # torch.save(learning.model, f"../{model_name}.pt")
    # plot_loss(train_loss_list, valid_loss_list, model_name)

    learning = LitMetricLearning(learning_rate=1e-5, weight_decay=0.1, model_name=model_name)
    logger = TensorBoardLogger("../", name="logs")
    trainer = L.Trainer(accelerator=dev, devices=1, max_epochs=20, logger=logger)
    trainer.fit(model=learning, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    # torch.save(learning.model, f"../{model_name}.pt")

    # checkpoint = "../logs/version_1/checkpoints/epoch_20.ckpt"
    # predictions, pred_labels, true_labels = predict(checkpoint, test_loader, torch.device(dev))
    #
    # print(accuracy_score(true_labels, pred_labels))
    # print(true_labels[:10])
    # print(pred_labels[:10])


if __name__ == "__main__":
    main()
