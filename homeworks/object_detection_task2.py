import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import sys
import torch
from torch.utils.data import random_split
from torchvision.models.feature_extraction import get_graph_node_names

sys.path
sys.path.append('../')

from torchvision import transforms
from src.object_detection import train_detect

def main():
    train_small_loader, valid_loader = train_val_split(train_images, train_labels, 0.3)
    train_loader = images_to_torch_dataset(train_images, train_labels)
    test_loader = images_to_torch_dataset(test_images, test_labels)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")