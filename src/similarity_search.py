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
from torch.utils.data import random_split
import torch.optim as optim
import torchvision.models.detection.backbone_utils
from torchvision import transforms, utils
from PIL import Image
from sklearn.metrics import accuracy_score

from torchvision.models import ResNet50_Weights, ResNet18_Weights
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

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
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=all_labels, cmap='tab10')
    plt.colorbar(scatter)
    plt.title("PCA of Embeddings")
    plt.savefig(f"../pca_embeddings.jpg")

    # # Reduce to 2D using t-SNE
    # tsne = TSNE(n_components=2, random_state=42)
    # embeddings_2d = tsne.fit_transform(all_embeddings)
    # plt.figure(figsize=(10, 10))
    # scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=all_labels, cmap='tab10')
    # plt.colorbar(scatter)
    # plt.title("t-SNE of Embeddings")
    # plt.savefig(f"../tsne_embeddings.jpg")


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
        print(self.data_table.shape)
        self.is_super_label = is_super_label

    def __getitem__(self, idx):
        image = load_image(self.imgs[idx])
        label = self.label[idx]
        super_label = self.super_label[idx]
        # print(self.imgs[idx])
        image_tensor = self.transform(image)
        out_label = super_label if self.is_super_label else label

        if self.is_train:
            if self.is_super_label:
                positive_list = self.imgs[(self.data_table.index != idx) & (self.super_label == super_label)].tolist()
                negative_list = self.imgs[(self.super_label != super_label)].tolist()
                # positive_image = load_image(self.data_table["pos_list_sup"][idx])
                # negative_image = load_image(self.data_table["neg_list_sup"][idx])
            else:
                positive_list = self.imgs[(self.data_table.index != idx) & (self.label == label)].tolist()
                negative_list = self.imgs[(self.super_label == super_label) & (self.label != label)].tolist()

            pos_path = random.choice(positive_list)
            positive_image = load_image(pos_path)
            neg_path = random.choice(negative_list)
            negative_image = load_image(neg_path)
            # print(pos_path, neg_path)

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
        # self.resnet = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.resnet = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        num_filters = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(nn.Linear(num_filters, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Linear(512, 128))

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
        n_labels = []
        train_loss = 0
        # for i, data in tqdm(enumerate(data_loader), desc=phase):
        for i, data in enumerate(data_loader):
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
            # # print(loss.item())
            # if phase == "valid":
            #     print(loss.item())
            train_loss += loss.item()

            if phase == "train":
                loss.backward()
                self.optimizer.step()
            true_labels.extend(anchor_label.to(torch.device("cpu")).detach().numpy())
            predictions.append(anchor_img_emb)
            n_labels.append(len(set(anchor_label.to(torch.device("cpu")).detach().numpy())))

        predictions = torch.cat(predictions, dim=0).to(torch.device("cpu")).detach().numpy()
        # print(f" mean number of classes per batch {np.mean(n_labels)}")
        # print(f" median number of classes per batch {np.median(n_labels)}")

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
        print_embeddings(predictions_train, true_labels_train)

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
            print(predictions.shape)

            for i, pred in enumerate(predictions):
                distances = (
                    self.triplet_loss.distance_function(pred, predictions).to(torch.device("cpu")).detach().numpy()
                )
                pred_im_idx = np.argsort(distances)[1]  #:6]

                pred_labels.append(np.take(true_labels, pred_im_idx))

        return predictions, pred_labels, true_labels


def load_data():
    train_data = f"{data_path}batch_train.csv"
    valid_data = f"{data_path}batch_val.csv"
    test_data = f"{data_path}batch_test.csv"
    transformers = [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        # transforms.GaussianBlur(kernel_size=(5, 9)),
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
    model_name = "model18_inference_batch_01_superlabel_batchnorm"

    learning = MetricLearning(learning_rate=0.01, weight_decay=0.1, model_name=model_name)

    # get data
    train_loader, valid_loader, test_loader = load_data()

    train_loss_list, valid_loss_list = learning.train_test(train_loader, valid_loader, num_epochs=200)
    torch.save(learning.model, f"../{model_name}.pt")
    plot_loss(train_loss_list, valid_loss_list, model_name)

    predictions, pred_labels, true_labels = learning.predict(test_loader)

    print(accuracy_score(true_labels, pred_labels))
    print(true_labels[:10])
    print(pred_labels[:10])


if __name__ == "__main__":
    # print(torch.backends.mps.is_available())
    main()
