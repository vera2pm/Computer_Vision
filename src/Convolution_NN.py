import torch.nn as nn
import torch
import torch.optim as optim
from torchvision.models import resnet50
from torchvision.models.feature_extraction import create_feature_extractor
import numpy as np

torch.backends.mps.is_available()


class CNN_MODEL(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=20, kernel_size=(7, 7))
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=40, kernel_size=(5, 5))
        self.fullconn1 = nn.Linear(in_features=40 * 1156, out_features=400)
        self.fullconn2 = nn.Linear(in_features=400, out_features=125)
        self.fullconn3 = nn.Linear(in_features=125, out_features=6)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.flatten(x)
        x = self.fullconn1(x)
        x = self.relu(x)

        x = self.fullconn2(x)
        x = self.relu(x)

        res = self.fullconn3(x)

        return res


class CNNFeatureExtractor(nn.Module):
    def __init__(self, nodes):
        super().__init__()
        m = resnet50()
        # Extract 4 main layers (note: MaskRCNN needs this particular name
        # mapping for return nodes)
        self.body = create_feature_extractor(
            m, return_nodes=nodes)

        inp = torch.randn(2, 3, 150, 150)
        with torch.no_grad():
            out = self.body(inp)
        in_channels_list = [o.shape[1] for o in out.values()]
        self.fullconn1 = nn.Linear(in_features=67500, out_features=400)
        self.fullconn2 = nn.Linear(in_features=400, out_features=125)
        self.fullconn3 = nn.Linear(in_features=125, out_features=6)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten(1)

    def forward(self, x):
        self.body(x)
        x = self.flatten(x)
        x = self.fullconn1(x)
        x = self.relu(x)
        x = self.fullconn2(x)
        x = self.relu(x)

        res = self.fullconn3(x)
        return res


def train_cnn(train_data, model, device, num_epochs=50, learning_rate=0.001, weight_decay=0.1):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model.to(device)
    train_loss_list = []
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}")
        train_loss = 0
        total = 0

        # Iterating over the training dataset in batches
        model.train()
        for i, (images, labels) in enumerate(train_data):
            # (images, labels) = train_data
            # Extracting images and target labels for the batch being iterated
            # now_batch_size = labels.size(0)
            images = images.to(device)
            labels = labels.to(device)

            # Calculating the model output and the cross entropy loss
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Updating weights according to calculated loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            total += labels.size(0)

        train_loss_list.append(train_loss / len(train_data))
        print(f"Training loss = {train_loss_list[-1]}")

    return model, train_loss_list


def test_cnn(test_data, model, device):
    test_acc = 0
    model.eval()
    total = 0
    with torch.no_grad():
        # Iterating over the training dataset in batches
        for i, (images, labels) in enumerate(test_data):
            images = images.to(device)
            y_true = labels.to(device)

            # Calculating outputs for the batch being iterated
            outputs = model(images)

            # Calculated prediction labels from models
            _, y_pred = torch.max(outputs.data, 1)

            # Comparing predicted and true labels
            test_acc += (y_pred == y_true).sum().item()
            total += labels.size(0)

        print(f"Test set accuracy = {test_acc / total * 100} % ")

    return model, test_acc, y_pred


if __name__ == "__main__":
    print(torch.backends.mps.is_available())