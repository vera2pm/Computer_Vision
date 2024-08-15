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
            pos_path = random.choice(positive_list)
            positive_image = load_image(pos_path)
            negative_list = self.imgs[(self.super_label == super_label) & (self.label != label)].tolist()
            neg_path = random.choice(negative_list)
            negative_image = load_image(neg_path)

            positive_tensor = self.transform(positive_image)
            negative_tensor = self.transform(negative_image)

            # print(self.imgs[idx], pos_path, neg_path)

            return image_tensor, label, positive_tensor, negative_tensor

        else:
            return image_tensor, label

    def __len__(self):
        return len(self.imgs)


# def get_clusters(X: np.ndarray, y: np.ndarray) -> List[np.ndarray]:
#     s = np.argsort(y)
#     return np.split(X[s], np.unique(y[s], return_index=True)[1][1:])


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
        self.resnet50.fc = nn.Sequential(nn.Linear(num_filters, 512), nn.ReLU(), nn.Linear(512, 10))
        # self.Triplet_loss = nn.Sequential(nn.Linear(10, 2))

    def forward(self, x):
        features = self.resnet50(x)
        # res = self.Triplet_loss(features)
        return features


def train_test(data_loader, model, device, num_epochs=50, learning_rate=0.001, weight_decay=0.1):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-7)
    triplet_loss = nn.TripletMarginWithDistanceLoss(margin=1.0)
    # triplet_loss = batch_all_triplet_loss()

    model.to(device)
    train_loss_list = []
    valid_loss_list = []
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}")
        train_loss = 0
        for phase in ["train", "valid"]:
            if phase == "train":
                model.train()
            else:
                model.eval()
            train_loss = 0
            total = 1
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
                positive_img_emb = model(positive_img)
                negative_img_emb = model(negative_img)

                loss = triplet_loss(anchor_img_emb, positive_img_emb, negative_img_emb)
                train_loss += loss.item()

                if phase == "train":
                    loss.backward()
                    optimizer.step()
                    # train_loss += loss.item()

                    # train_loss_list.append(train_loss)  # / len(data))
                    # if i % 300 == 0:
                    #     print(f"Training loss = {train_loss_list[-1]}")
                total += i

            if phase == "train":
                train_loss_list.append(train_loss / total)
                print(f"Training loss = {train_loss_list[-1]}")
            else:
                valid_loss_list.append(train_loss / total)
                print(f"Valid loss = {valid_loss_list[-1]}")

    # torch.save(model, "../similarity_model_inference.pt")

    return model, train_loss_list, valid_loss_list


def predict_batch(test_loader, model, device):
    model = model.to(device)
    pred_labels = []
    predictions = []
    true_labels = []
    for data in test_loader:
        test_images, test_labels = data
        test_images, test_labels = test_images.to(device), test_labels.to(device)

        test_embs = model(test_images)
        distances = _pairwise_distances(test_embs).to(torch.device("cpu")).detach().numpy()
        # print(distances)
        # for i, (test_img, labels) in enumerate(test_loader):
        for i, preds in enumerate(distances):
            pred_im_idx = np.argsort(preds)[1]
            # print(preds[pred_im_idx])
            true_labels.append(int(test_labels[i].to(torch.device("cpu")).detach()))
            pred_labels.append(int(test_labels[pred_im_idx].to(torch.device("cpu")).detach()))
            # predictions.append(test_images[pred_im_idx].to(torch.device("cpu")).detach().numpy())

    return predictions, pred_labels, true_labels


def predict(test_loader, model, device):
    model = model.to(device)
    pred_labels = []
    predictions = None
    true_labels = []
    for data in test_loader:
        test_images, test_labels = data
        test_images, test_labels = test_images.to(device), test_labels.to(device)

        test_embs = model(test_images)

        true_labels.extend(test_labels.to(torch.device("cpu")).detach().numpy())
        pred_labels.append(test_labels.to(torch.device("cpu")).detach().numpy())
        if predictions is None:
            predictions = test_embs
        else:
            print(test_embs.shape)
            print(predictions.shape)
            predictions = torch.cat((predictions, test_embs), dim=-1)  # breaking bc of memory here
            print(predictions.shape)
            # predictions.append(test_embs.to(torch.device("cpu")).detach().numpy())
    print(predictions)
    distances = _pairwise_distances(predictions)
    print(distances)
    # for i, (test_img, labels) in enumerate(test_loader):
    # for i, preds in enumerate(distances):
    #     pred_im_idx = np.argsort(preds)[1]
    # print(preds[pred_im_idx])
    # true_labels.append(int(test_labels[i].to(torch.device("cpu")).detach()))
    # pred_labels.append(int(test_labels[pred_im_idx].to(torch.device("cpu")).detach()))
    # predictions.append(test_images[pred_im_idx].to(torch.device("cpu")).detach().numpy())

    return predictions, pred_labels, true_labels


def load_data():
    train_data = f"{data_path}Ebay_train_preproc.csv"
    test_data = f"{data_path}Ebay_test_preproc.csv"
    transformers = [transforms.Resize((400, 400)), transforms.ToTensor()]

    train_data_ds = TripletDataset(train_data, is_train=True, transform=transformers)
    train_cb_sampler = CustomBatchSampler(train_data_ds.super_label, train_data_ds.label, batch_size=4)
    train_loader = torch.utils.data.DataLoader(train_data_ds, batch_sampler=train_cb_sampler)

    test_data_ds = TripletDataset(test_data, is_train=False, transform=transformers)
    # test_cb_sampler = CustomBatchSampler(test_data_ds.super_label, test_data_ds.label, batch_size=4)
    test_loader = torch.utils.data.DataLoader(test_data_ds, batch_size=10)
    return train_loader, test_loader


def plot_loss(train_loss_list, valid_loss_list, model_name):
    df = pd.DataFrame(columns=["train", "valid"])
    df["train"] = train_loss_list
    df["valid"] = valid_loss_list
    sns.lineplot(data=df)
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.savefig(f"../{model_name}.jpg")


def main():
    model = ResnetTriplet()

    if torch.backends.mps.is_available():
        dev = "mps"
    elif torch.cuda.is_available():
        dev = "cuda"
    else:
        dev = "cpu"
    device = torch.device(dev)
    print(f"device: {device}")

    # get data
    train_loader, test_loader = load_data()

    model, train_loss_list, valid_loss_list = train_test(
        train_loader, model, device, num_epochs=50, learning_rate=0.001, weight_decay=0.1
    )
    print(train_loss_list)
    model_name = "model18_inference"
    torch.save(model, f"../{model_name}.pt")
    plot_loss(train_loss_list, valid_loss_list, model_name)

    model = torch.load(f"../{model_name}.pt")

    predictions, pred_labels, true_labels = predict(test_loader, model, device)

    print(accuracy_score(true_labels, pred_labels))


if __name__ == "__main__":
    # print(torch.backends.mps.is_available())
    main()
