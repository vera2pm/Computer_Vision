import random
from typing import List, Optional

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import random_split
import torch.optim as optim
import torchvision.models.detection.backbone_utils
from torchvision import transforms, utils
from PIL import Image

from online_triplet_loss.losses import batch_all_triplet_loss, _pairwise_distances


class TripletDataset(torch.utils.data.Dataset):

    def __init__(self, data_path: str, is_train: bool = True, transform: Optional[List] = None):
        self.data_table = pd.read_table(data_path)
        self.imgs = self.data_table[3]
        self.label = self.data_table[1]
        self.super_label = self.data_table[2]
        self.is_train = is_train
        # if transform is None:
        #     transform = list()
        self.transform = transforms.Compose(transform.append(transforms.ToTensor()))

    def __getitem__(self, idx):
        image = Image.open(self.imgs[idx])
        label = self.label[idx]
        super_label = self.super_label[idx]

        image_tensor = self.transform(image)

        if self.is_train:
            positive_list = self.data_table[(self.data_table.index != idx) & (self.label == label)][3]
            positive_image = Image.open(random.choice(positive_list))
            negative_list = self.data_table[(self.super_label == super_label) & (self.label != label)][3]
            negative_image = Image.open(random.choice(negative_list))

            positive_tensor = self.transform(positive_image)
            negative_tensor = self.transform(negative_image)

            return image_tensor, label, positive_tensor, negative_tensor

        else:

            return image_tensor, label

    def __len__(self):
        return len(self.imgs)


class ResnetTriplet(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet50 = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        num_filters = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Sequential(nn.Linear(num_filters, 512), nn.ReLU(), nn.Linear(512, 10))
        # self.Triplet_loss = nn.Sequential(nn.Linear(10, 2))

    def forward(self, x):
        features = self.resnet50(x)
        # res = self.Triplet_loss(features)
        return features


def train_test(data_loader, model, device, num_epochs=50, learning_rate=0.001, weight_decay=0.1):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-7)
    # triplet_loss = batch_all_triplet_loss()

    model.to(device)
    train_loss_list = []
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}")
        train_loss = 0
        for phase in ["train", "valid"]:
            if phase == "train":
                model.train()
            else:
                model.eval()
            # Iterating over the training dataset in batches
            for i, data in enumerate(data_loader):
                # Extracting images and target labels for the batch being iterated
                anchor_img, anchor_label, positive_img, negative_img = data
                anchor_img, anchor_label, positive_img, negative_img = (
                    anchor_img.to(device),
                    anchor_label.to(device),
                    positive_img.to(device),
                    negative_img.to(device),
                )

                optimizer.zero_grad()

                anchor_img_emb = model(anchor_img)

                if phase == "train":
                    positive_img_emb = model(positive_img)
                    negative_img_emb = model(negative_img)

                    loss = triplet_loss(anchor_img_emb, positive_img_emb, negative_img_emb)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()

                    # train_loss_list.append(train_loss / len(train_data))
                    # print(f"Training loss = {train_loss_list[-1]}")

                else:
                    continue

    return model, train_loss_list


def predict(test_loader, model, device):
    model = model.to(device)
    test_images, test_labels = test_loader
    test_images, test_labels = test_images.to(device), test_labels.to(device)

    test_embs = model(test_images)
    distances = _pairwise_distances(test_embs)
    pred_labels = []
    predictions = []
    for i, test_img, labels in test_loader:
        pred_im_idx = np.argmin(distances[i])
        pred_labels.append(test_labels[pred_im_idx])
        predictions.append(test_images[pred_im_idx])

    return predictions, pred_labels


def load_data():
    train_data = "../data/Stanford_Online_Products/Ebay_train.txt"
    test_data = "../data/Stanford_Online_Products/Ebay_test.txt"
    transformers = [transforms.Resize((760, 1140))]

    train_data_ds = TripletDataset(train_data, is_train=True, transform=transformers)
    train_loader = torch.utils.data.DataLoader(train_data_ds, batch_size=5, shuffle=True)

    test_data_ds = TripletDataset(test_data, is_train=False, transform=transformers)
    test_loader = torch.utils.data.DataLoader(test_data_ds, batch_size=5, shuffle=False)
    return train_loader, test_loader


def main():
    model = ResnetTriplet()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    # get data
    train_loader, test_loader = load_data()

    model, train_loss_list = train_test(
        train_loader, model, device, num_epochs=5, learning_rate=0.001, weight_decay=0.1
    )

    predictions, pred_labels = predict(test_loader, model, device)
