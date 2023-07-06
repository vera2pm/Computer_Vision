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
from src.utils import find_homo, load_data, plot_5images, cv2_load2rgb
from src.configs.classification_config import CLASS_NAMES, CLASS_NAMES_LABEL
from src.knn_classification import KnnClassificate
from torchvision.models import AlexNet, resnet50
from src.Convolution_NN import CNN_MODEL, train_cnn, test_cnn, CNNFeatureExtractor, train_val


def images_to_torch_dataset(train_images, train_labels):
    transform = transforms.ToTensor()
    train_images = np.array(train_images, dtype="float32")
    train_images_tensors = list(map(transform, train_images))
    train_data = torch.utils.data.TensorDataset(
        torch.stack(train_images_tensors),
        torch.from_numpy(np.array(train_labels)))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=40, shuffle=True)
    return train_loader


def train_val_split(train_images, train_labels, val_ratio):
    transform = transforms.ToTensor()
    train_images = np.array(train_images, dtype="float32")
    train_images_tensors = list(map(transform, train_images))
    dataset = torch.utils.data.TensorDataset(
        torch.stack(train_images_tensors),
        torch.from_numpy(np.array(train_labels)))
    batch_size = 40
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size

    train_data, val_data = random_split(dataset, [train_size, val_size])
    print(f"Length of Train Data : {len(train_data)}")
    print(f"Length of Validation Data : {len(val_data)}")

    # load the train and validation into batches.
    train_dl = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dl = torch.utils.data.DataLoader(val_data, batch_size * 2, num_workers=4, pin_memory=True)

    return train_dl, val_dl


def plot_train_val(trainval_loss_list, val_acc_list, title="CNN local model"):
    fig = plt.figure(figsize=(15,5))
    ax0 = fig.add_subplot(121, title="Train Loss", xlabel="Epoch")
    ax1 = fig.add_subplot(122, title="Accuracy", xlabel="Epoch")
    sns.lineplot(trainval_loss_list["train"], ax=ax0);
    sns.lineplot(val_acc_list, ax=ax1);
    fig.suptitle(title);
    plt.savefig(f'../{title}.jpg')


def main():
    nb_classes = len(CLASS_NAMES)
    IMAGE_SIZE = (150, 150)
    # prepare dataset
    print("prepare dataset")
    datasets_path = ["../Classification_data/train/", "../Classification_data/test/"]
    (train_images, train_labels), (test_images, test_labels) = load_data(datasets_path, CLASS_NAMES_LABEL, IMAGE_SIZE)
    print(len(train_images))
    print(len(test_images))
    # plot_5images(train_images[:5])

    # KNN
    # print("KNN")
    # knn = KnnClassificate(k=3, image_size=150*150*3)
    # knn.train(train_images, train_labels)
    # predict_test_labels = knn.predict(test_images)
    # knn.evaluate_model(test_labels, predict_test_labels)

    # CNN
    print("CNN")
    # prepare torch dataset
    train_small_loader, valid_loader = train_val_split(train_images, train_labels, 0.3)
    train_loader = images_to_torch_dataset(train_images, train_labels)
    test_loader = images_to_torch_dataset(test_images, test_labels)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    # model CNN
    print("CNN")
    model = CNN_MODEL(num_channels=3)
    # check parameters
    model, trainval_loss_list, val_acc_list = train_val(
        train_small_loader, valid_loader, model, device, num_epochs=10, learning_rate=1.0e-4, weight_decay=0.1)
    plot_train_val(trainval_loss_list, val_acc_list, title="CNN local model")
    # train
    model, train_loss_list = train_cnn(
        train_loader, model, device, num_epochs=10, learning_rate=1.0e-4, weight_decay=0.1)
    # test
    model, test_acc, y_pred = test_cnn(test_loader, model, device)

    # model AlexNet
    print("AlexNet")
    model_alex = AlexNet(num_classes=6)
    # check parameters
    model_alex, trainval_loss_list_alex, val_acc_list_alex = train_val(
        train_small_loader, valid_loader, model_alex, device, num_epochs=10, learning_rate=1.0e-4, weight_decay=0.1)
    plot_train_val(trainval_loss_list_alex, val_acc_list_alex, title="ALEX model")
    # train
    model_alex, train_loss_list_alex = train_cnn(
        train_loader, model_alex, device, num_epochs=10, learning_rate=1.0e-4, weight_decay=0.1)
    # test
    model_alex, test_acc_alex, y_pred_alex = test_cnn(test_loader, model_alex, device)

    # model with feature extractor
    print("Feature Extractor")
    train_nodes, eval_nodes = get_graph_node_names(resnet50())
    return_nodes = train_nodes[5:-2]
    model_features = CNNFeatureExtractor(return_nodes)

    # check parameters
    model_features, trainval_loss_list_feat, val_acc_list_feat = train_val(
        train_small_loader, valid_loader, model_features, device, num_epochs=10, learning_rate=1.0e-4, weight_decay=0.1)
    plot_train_val(trainval_loss_list_feat, val_acc_list_feat, title="Model with Feature Extraction")
    # train
    print("\n--------  \n  ----------\n")
    model_features, train_loss_list_feat = train_cnn(
        train_loader, model_features, device, num_epochs=10, learning_rate=1.0e-4, weight_decay=0.1)
    # test
    model_features, test_acc_feat, y_pred_feat = test_cnn(test_loader, model_features, device)

    df = pd.DataFrame.from_dict(
        {"original": train_loss_list, "alex": train_loss_list_alex, "feature_ex": train_loss_list_feat})

    fig = sns.lineplot(df)
    plt.xlabel("epoch")
    plt.ylabel("train loss")
    plt.savefig('../train_compare10.jpg')


if __name__ == "__main__":
    print(torch.backends.mps.is_available())
    main()
