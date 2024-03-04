import torchvision.models.detection.backbone_utils
import torch
from torch.utils.data import random_split
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from matplotlib import pyplot as plt

from src.object_detection import train_detect, ObjDetectAnimalDataset, MyCompose, Resize, collate_fn, eval_detect


def main():
    data_1 = torch.load("../data/object_detect/train_set_1.pt")
    data_2 = torch.load("../data/object_detect/train_set_2.pt")
    train_data = data_1[:-2] + data_2[:-2]
    test_data = data_1[-2:] + data_2[-2:]

    data = torch.load("../whole_set_5.pt")
    train_data = data[:-10]
    test_data = data[-10:]
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

    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    transforms_train = MyCompose([Resize((760, 1140))])
    train_data = ObjDetectAnimalDataset(train_data, transforms=transforms_train)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=5, shuffle=True, collate_fn=collate_fn)
    model, train_loss_list = train_detect(
        train_loader, model, device, num_epochs=5, learning_rate=0.001, weight_decay=0.1
    )

    transforms_test = MyCompose([Resize((760, 1140))])
    test_data_ds = ObjDetectAnimalDataset(test_data, transforms=transforms_test)
    test_loader = torch.utils.data.DataLoader(test_data_ds, batch_size=5, shuffle=False, collate_fn=collate_fn)

    predictions, metric, test_loss_list = eval_detect(test_loader, model, device)
    metric.compute()
    fig_, ax_ = metric.plot()
    plt.savefig("../eval_object_detection_plot.jpg")

    metric.plot(test_loss_list)
    plt.savefig("../eval_object_detection_test_loss.jpg")

    torch.save(predictions, f"predictions.pt")


if __name__ == "__main__":
    print(torch.backends.mps.is_available())
    main()
