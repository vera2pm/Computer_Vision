import torch.nn as nn
import torch.optim as optim

class CNN:
    def __init__(self, num_channels):
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=20, kernel_size=(5, 5))
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=(5, 5))
        self.fc = nn.Linear(in_features=800, out_features=500)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.relu = nn.ReLU()

        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = nn.Flatten(x, 1)
        x = self.fc(x)
        x = self.relu(x)

        res = self.logSoftmax(x)
        return res

    def train(self):
        pass
