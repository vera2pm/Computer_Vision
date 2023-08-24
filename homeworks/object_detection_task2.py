import numpy as np
import pandas as pd
import seaborn as sns
import torchvision.models.detection.backbone_utils
from matplotlib import pyplot as plt
import sys
import torch
from torch.utils.data import random_split
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.feature_extraction import get_graph_node_names

from torchvision import transforms
from src.object_detection import train_detect, ObjDetectAnimalDataset, MyCompose, Resize, collate_fn


def main():
    data_1 = torch.load("../data/object_detect/train_set_1.pt")
    data_2 = torch.load("../data/object_detect/train_set_2.pt")
    train_data = data_1[:-2] + data_2[:-2]
    test_data = data_1[-2:] + data_2[-2:]
    print(len(train_data))
    print(len(test_data))

    # train_small_loader, valid_loader = train_val_split(train_images, train_labels, 0.3)
    # train_loader = images_to_torch_dataset(train_images, train_labels)
    # test_loader = images_to_torch_dataset(test_images, test_labels)

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes = 3
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # print(model)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device("cpu")
    print(f"device: {device}")

    transforms_train = MyCompose([Resize((760, 1140))])
    train_data = ObjDetectAnimalDataset(train_data, transforms=transforms_train)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=5, shuffle=True, collate_fn=collate_fn)
    model, train_loss_list = train_detect(train_loader, model, device, num_epochs=5, learning_rate=0.001, weight_decay=0.1)


if __name__ == "__main__":
    print(torch.backends.mps.is_available())
    main()
