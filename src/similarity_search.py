import random
from typing import List, Optional
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import random_split, Sampler
import torch.optim as optim
import torchvision.models.detection.backbone_utils
from torchvision import transforms, utils
from PIL import Image
from sklearn.metrics import accuracy_score

from online_triplet_loss.losses import batch_all_triplet_loss, _pairwise_distances
from torchvision.models import ResNet50_Weights, ResNet18_Weights

data_path = "../data/Stanford_Online_Products/"


def load_image(image_path):
    image = Image.open(data_path + image_path)
    return image


class TripletDataset(torch.utils.data.Dataset):
    def __init__(self, data_path: str, is_train: bool = True, transform: Optional[List] = None):
        self.data_table = pd.read_csv(data_path)
        self.imgs = self.data_table["path"]
        self.label = self.data_table["class_id"]
        self.super_label = self.data_table["super_class_id"]
        self.is_train = is_train
        # if transform is None:
        #     transform = list()
        # transform.append(transforms.ToTensor())
        self.transform = transforms.Compose(transform)

    def __getitem__(self, idx):
        image = load_image(self.imgs[idx])
        label = self.label[idx]
        super_label = self.super_label[idx]
        image_tensor = self.transform(image)

        if self.is_train:
            positive_list = self.imgs[(self.data_table.index != idx) & (self.label == label)].tolist()
            # positive_list = self.imgs[(self.data_table.index != idx) & (self.super_label == super_label)].tolist()
            pos_path = random.choice(positive_list)
            positive_image = load_image(pos_path)
            negative_list = self.imgs[(self.super_label == super_label) & (self.label != label)].tolist()
            # negative_list = self.imgs[(self.super_label != super_label)].tolist()
            neg_path = random.choice(negative_list)
            negative_image = load_image(neg_path)

            positive_tensor = self.transform(positive_image)
            negative_tensor = self.transform(negative_image)

            return image_tensor, label, positive_tensor, negative_tensor

        else:
            return image_tensor, label

    def __len__(self):
        return len(self.imgs)


class CustomBatchSampler(Sampler):
    r"""Yield a mini-batch of indices. The sampler will drop the last batch of
            an image size bin if it is not equal to ``batch_size``

    Args:
        examples (dict): List from dataset class.
        batch_size (int): Size of mini-batch.
    """

    def __init__(self, super_labels, labels, batch_size):
        self.batch_size = batch_size
        self.data = {}
        for i, (label, super_label) in enumerate(zip(labels, super_labels)):
            if super_label in self.data:
                self.data[super_label].append((i, label))
            else:
                self.data[super_label] = [(i, label)]

        self.total = 0
        for label, indexes in self.data.items():
            self.total += len(indexes) // self.batch_size

    def __iter__(self):
        for _, indexes_labels in self.data.items():
            count = len(indexes_labels)
            batch = []
            batch_labels = {}
            labels = []
            last_batch = []

            for i, (idx, label) in enumerate(indexes_labels):
                batch.append(idx)
                labels.append(label)
                batch_labels[label] = batch_labels[label] + 1 if label in batch_labels.keys() else 1

                if i == count - 1 and len(batch) < self.batch_size:
                    length = np.ceil((self.batch_size - len(batch)) / 2) * 2
                    batch.extend(last_batch[-length:])

                if len(batch) >= self.batch_size:
                    if np.min(list(batch_labels.values())) > 1:
                        yield batch
                        batch = []
                        labels = []
                        batch_labels = {}
                        last_batch = batch
                    else:
                        continue

    def __len__(self):
        return self.total


class ResnetTriplet(nn.Module):
    def __init__(self):
        super().__init__()
        # self.resnet50 = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.resnet50 = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        num_filters = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Sequential(nn.Linear(num_filters, 512), nn.ReLU(), nn.Linear(512, 128))

    def forward(self, x):
        features = self.resnet50(x)
        return features


class MetricLearning:
    def __init__(self, learning_rate, weight_decay, model_name=None):
        if model_name is None:
            self.model = ResnetTriplet()
        else:
            self.model = torch.load(f"../{model_name}.pt")

        if torch.backends.mps.is_available():
            dev = "mps"
        elif torch.cuda.is_available():
            dev = "cuda"
        else:
            dev = "cpu"
        self.device = torch.device(dev)
        print(f"device: {self.device}")

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.triplet_loss = nn.TripletMarginWithDistanceLoss(margin=1.0)
        # self.triplet_loss = batch_all_triplet_loss()

    def calculate_loss(self, data_loader, phase):
        predictions = []
        true_labels = []
        train_loss = 0
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
                    train_loss, _, _ = self.calculate_loss(train_loader, "train")
                    train_loss_list.append(train_loss / len(train_loader))
                    print(f"Training loss = {train_loss_list[-1]}")
                else:
                    with torch.no_grad():
                        self.model.eval()
                        # Iterating over the training dataset in batches
                        valid_loss, predictions, true_labels = self.calculate_loss(valid_loader, "valid")

                        valid_loss_list.append(valid_loss / len(valid_loader))
                        print(f"Valid loss = {valid_loss_list[-1]}")

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
            N = predictions.shape[0]

            for i, pred in enumerate(predictions):
                distances = (
                    self.triplet_loss.distance_function(pred, predictions).to(torch.device("cpu")).detach().numpy()
                )

                pred_im_idx = np.argsort(distances)[1]
                pred_labels.append(true_labels[pred_im_idx])

        return predictions, pred_labels, true_labels


# def predict_batch(test_loader, model, device):
#     model = model.to(device)
#     model.eval()
#     pred_labels = []
#     predictions = []
#     true_labels = []
#     for data in test_loader:
#         test_images, test_labels = data
#         test_images, test_labels = test_images.to(device), test_labels.to(device)
#
#         test_embs = model(test_images)
#         distances = _pairwise_distances(test_embs).to(torch.device("cpu")).detach().numpy()
#         # print(distances)
#         # for i, (test_img, labels) in enumerate(test_loader):
#         for i, preds in enumerate(distances):
#             pred_im_idx = np.argsort(preds)[1]
#             # print(preds[pred_im_idx])
#             true_labels.append(int(test_labels[i].to(torch.device("cpu")).detach()))
#             pred_labels.append(int(test_labels[pred_im_idx].to(torch.device("cpu")).detach()))
#             # predictions.append(test_images[pred_im_idx].to(torch.device("cpu")).detach().numpy())
#
#     return predictions, pred_labels, true_labels


def load_data():
    train_data = f"{data_path}Ebay_train_train_preproc.csv"
    valid_data = f"{data_path}Ebay_train__val_preproc.csv"
    test_data = f"{data_path}Ebay_test_preproc.csv"
    transformers = [transforms.Resize((400, 400)), transforms.ToTensor()]

    train_data_ds = TripletDataset(train_data, is_train=True, transform=transformers)
    train_cb_sampler = CustomBatchSampler(train_data_ds.super_label, train_data_ds.label, batch_size=4)
    train_loader = torch.utils.data.DataLoader(train_data_ds, batch_sampler=train_cb_sampler)

    valid_data_ds = TripletDataset(valid_data, is_train=True, transform=transformers)
    valid_cb_sampler = CustomBatchSampler(valid_data_ds.super_label, valid_data_ds.label, batch_size=4)
    valid_loader = torch.utils.data.DataLoader(valid_data_ds, batch_sampler=valid_cb_sampler)

    test_data_ds = TripletDataset(test_data, is_train=False, transform=transformers)
    test_cb_sampler = CustomBatchSampler(test_data_ds.super_label, test_data_ds.label, batch_size=10)
    test_loader = torch.utils.data.DataLoader(test_data_ds, batch_size=20)
    # test_loader = torch.utils.data.DataLoader(test_data_ds, batch_sampler=test_cb_sampler)

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
    # model = ResnetTriplet()
    #
    # if torch.backends.mps.is_available():
    #     dev = "mps"
    # elif torch.cuda.is_available():
    #     dev = "cuda"
    # else:
    #     dev = "cpu"
    # device = torch.device(dev)
    # print(f"device: {device}")
    model_name = "model18_inference_n128_super_label"

    learning = MetricLearning(learning_rate=0.001, weight_decay=0.1)
    # learning = MetricLearning(learning_rate=0.001, weight_decay=0.1, model_name=model_name)

    # get data
    train_loader, valid_loader, test_loader = load_data()

    train_loss_list, valid_loss_list = learning.train_test(train_loader, valid_loader, num_epochs=50)
    print(train_loss_list)
    torch.save(learning.model, f"../{model_name}.pt")
    plot_loss(train_loss_list, valid_loss_list, model_name)

    predictions, pred_labels, true_labels = learning.predict(test_loader)

    print(accuracy_score(true_labels, pred_labels))
    print(len(true_labels))
    print(len(pred_labels))


if __name__ == "__main__":
    # print(torch.backends.mps.is_available())
    main()
