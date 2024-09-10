import random
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


class LitMetricLearning(L.LightningModule):
    def __init__(self, learning_rate, weight_decay):
        super().__init__()
        # self.model_path = f"../{model_name}.pt"
        # if os.path.exists(self.model_path):
        #     print("Load existing model")
        #     self.model = torch.load(self.model_path)
        # else:
        #     print("Train new model")
        self.model = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        num_filters = self.model.fc.in_features
        self.model.fc = nn.Sequential(nn.Linear(num_filters, 512), nn.ReLU(), nn.Linear(512, 128))

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.triplet_loss = nn.TripletMarginWithDistanceLoss(margin=1.0, swap=True)

    def forward(self, x):
        features = self.model(x)
        features = F.normalize(features, p=2, dim=1)
        return features

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

    def predict_step(self, batch, batch_idx):
        test_images, test_labels = batch
        test_embs = self.model(test_images)
        return test_embs, test_labels


def find_similar_items(predictions_labels, model):
    true_labels = []
    predictions = []
    for preds, test_labels in tqdm(predictions_labels):
        predictions.append(preds)
        true_labels.append(test_labels)

    predictions = torch.cat(predictions, dim=0)
    true_labels = torch.cat(true_labels, dim=0)

    pred_labels = []
    for i, pred in tqdm(enumerate(predictions)):
        distances = model.triplet_loss.distance_function(pred, predictions).to(torch.device("cpu")).detach().numpy()
        pred_im_idx = np.argsort(distances)[1]
        pred_labels.append(np.take(true_labels, pred_im_idx))

    print(accuracy_score(true_labels, pred_labels))
    print(true_labels[:10])
    print(pred_labels[:10])


def load_data():
    train_data = f"{data_path}Ebay_train_train_preproc.csv"
    valid_data = f"{data_path}Ebay_train__val_preproc.csv"
    test_data = f"{data_path}Ebay_train__val_preproc.csv"
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

    # get data
    train_loader, valid_loader, test_loader = load_data()

    learning = LitMetricLearning(learning_rate=1e-5, weight_decay=0.1)
    logger = TensorBoardLogger("../", name="logs")
    trainer = L.Trainer(accelerator=dev, devices=1, max_epochs=20, logger=logger)
    trainer.fit(model=learning, train_dataloaders=train_loader, val_dataloaders=valid_loader)

    predictions = trainer.predict(learning, dataloaders=test_loader)
    find_similar_items(predictions, learning)


if __name__ == "__main__":
    main()
