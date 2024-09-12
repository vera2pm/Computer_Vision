import numpy as np
from typing import Optional, List, Any

import lightning as L
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

import torch
from torch.utils.data import random_split
import torch.optim as optim
from torchmetrics.detection import MeanAveragePrecision
from torchvision.transforms import functional as F
import torchvision.models.detection.backbone_utils
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from src.utils import get_device


def collate_fn(batch):
    return tuple(zip(*batch))


class MyCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, tar):
        for t in self.transforms:
            img, tar = t(img, tar)
        return img, tar


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        y_ = image.shape[1]
        x_ = image.shape[2]
        ysize, xsize = self.size
        x_scale = xsize / x_
        y_scale = ysize / y_
        image = F.resize(image, self.size)
        new_boxes = []
        for box in np.array(target["boxes"]).tolist():
            (origx, origy, origxmax, origymax) = box
            x = origx * x_scale
            y = origy * y_scale
            xmax = origxmax * x_scale
            ymax = origymax * y_scale
            new_boxes.append([x, y, xmax, ymax])
        target["boxes"] = torch.tensor(new_boxes, dtype=torch.float32)
        return image, target


class ObjDetectAnimalDataset(torch.utils.data.Dataset):
    def __init__(self, imgs, transform: Optional[List] = None):
        self.transforms = MyCompose(transform)
        self.imgs = imgs

    def __getitem__(self, idx):
        data = self.imgs[idx]
        image, target = data

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.imgs)


class ObjectDetectDataModule(L.LightningDataModule):
    def __init__(self, train_data: str, test_data: str, transformers: list, batch_size: int):
        super().__init__()
        self.train_data = train_data
        self.test_data = test_data
        self.batch_size = batch_size
        self.transformers = transformers

    def setup(self, stage: str) -> None:
        if stage == "fit":
            train_data = ObjDetectAnimalDataset(self.train_data, transform=self.transformers)
            self.train_subset, self.val_subset = random_split(
                train_data, [0.8, 0.2], generator=torch.Generator().manual_seed(42)
            )
        if stage == "test":
            self.test_ds = ObjDetectAnimalDataset(self.test_data, transform=self.transformers)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return torch.utils.data.DataLoader(
            self.train_subset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=7,
            persistent_workers=True,
            collate_fn=collate_fn,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return torch.utils.data.DataLoader(
            self.val_subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=7,
            persistent_workers=True,
            collate_fn=collate_fn,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return torch.utils.data.DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=7,
            persistent_workers=True,
            collate_fn=collate_fn,
        )


# def train_detect(train_data, model, device, num_epochs=50, learning_rate=0.001, weight_decay=0.1):
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
#
#     model.to(device)
#     train_loss_list = []
#     for epoch in range(num_epochs):
#         print(f"Epoch {epoch}")
#         train_loss = 0
#         total = 0
#
#         # Iterating over the training dataset in batches
#         model.train()
#         for i, (images, targets) in enumerate(train_data):
#             # Extracting images and target labels for the batch being iterated
#             # print(targets)
#             images = list(image.to(device) for image in images)
#             targets = [
#                 {key: v.to(device) if isinstance(v, torch.Tensor) else v for key, v in t.items()} for t in targets
#             ]
#
#             # Calculating the model output and the cross entropy loss
#             loss_dict = model(images, targets)
#             losses = sum(loss for loss in loss_dict.values())
#
#             # Updating weights according to calculated loss
#             optimizer.zero_grad()
#             losses.backward()
#             optimizer.step()
#             train_loss += losses.item()
#             total += len(targets)
#
#         train_loss_list.append(train_loss / len(train_data))
#         print(f"Training loss = {train_loss_list[-1]}")
#
#     return model, train_loss_list
#
#
# def eval_detect(test_data, model, device):
#     model.to(device)
#     test_loss_list = []
#     model.eval()
#     metric = MeanAveragePrecision(iou_type="bbox")
#     for i, (images, targets) in enumerate(test_data):
#         images = list(img.to(device) for img in images)
#
#         predictions = model(images)
#         predictions = [{k: v.to(device) for k, v in t.items()} for t in predictions]
#
#         targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
#         print(targets)
#
#         # gather the stats from all processes
#         test_loss_list.append(metric(predictions, targets))
#
#     # torch.set_num_threads(n_threads)
#     return predictions, metric, test_loss_list


class ObjectDetectionModule(L.LightningModule):
    def __init__(self, model, learning_rate, weight_decay):
        super().__init__()
        self.model = model
        self.save_hyperparameters(ignore=["model"])

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.metric = MeanAveragePrecision(iou_type="bbox")

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer

    def _calculate_loss_per_batch(self, data):
        images, targets = data
        loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        return losses

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        loss = self._calculate_loss_per_batch(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        loss = self._calculate_loss_per_batch(batch)
        self.log("val_loss", loss, prog_bar=True)

    def test_step(self, batch, *args: Any, **kwargs: Any):
        images, targets = batch
        predictions = self.model(images)

        metric = self.metric(predictions, targets)
        self.log("test_map", metric, prog_bar=True)

    def predict_step(self, batch, batch_idx):
        images, targets = batch
        predictions = self.model(images)
        return predictions


def main():
    data_1 = torch.load("../data/object_detect/train_set_1.pt")
    data_2 = torch.load("../data/object_detect/train_set_2.pt")
    train_data = data_1[:-2] + data_2[:-2]
    test_data = data_1[-2:] + data_2[-2:]

    # data = torch.load("../whole_set_5.pt")
    # train_data = data[:-10]
    # test_data = data[-10:]
    # print(len(train_data))
    # print(len(test_data))

    model_name = "ObjDetect"
    dev = get_device()

    logger = TensorBoardLogger("../logs/", name=model_name)
    checkpoint_callback = ModelCheckpoint(dirpath=f"../logs/{model_name}/", save_top_k=1, monitor="val_loss")
    trainer = L.Trainer(accelerator=dev, devices=1, max_epochs=1, logger=logger, callbacks=[checkpoint_callback])

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes = 3
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    transformers = [Resize((224, 224))]
    data_module = ObjectDetectDataModule(train_data, test_data, transformers, 5)

    learning = ObjectDetectionModule(model, learning_rate=0.001, weight_decay=0.1)
    trainer.fit(model=learning, datamodule=data_module)

    trainer.test(learning, datamodule=data_module)

    # metric.compute()
    # fig_, ax_ = metric.plot()
    # plt.savefig("../eval_object_detection_plot.jpg")
    #
    # metric.plot(test_loss_list)
    # plt.savefig("../eval_object_detection_test_loss.jpg")
    #
    # torch.save(predictions, f"predictions.pt")


if __name__ == "__main__":
    print(torch.backends.mps.is_available())
    main()
