from typing import Optional, List, Any
import os
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
import torch
from torch.utils.data import random_split
from torchvision.models.feature_extraction import get_graph_node_names

from src.utils import get_device

from torchvision import transforms
from src.configs.classification_config import CLASS_NAMES_LABEL, IMAGE_SIZE
from torchvision.models import AlexNet, resnet50
from src.classification_models.Convolution_NN import CNNFeatureExtractor, CNNModel
import torch.optim as optim
import torch.nn as nn
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from PIL import Image

import lightning as L
from tqdm import tqdm


class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, data_path: str, transform: Optional[List] = None):
        self.transform = transforms.Compose(transform)
        self.imgs = []
        self.label = []
        self.class_names_label = CLASS_NAMES_LABEL

        for folder in os.listdir(data_path):
            if folder not in self.class_names_label.keys():
                continue
            label = self.class_names_label[folder]
            for file in tqdm(os.listdir(os.path.join(data_path, folder))):
                # Get the path name of the image
                self.imgs.append(os.path.join(os.path.join(data_path, folder), file))
                self.label.append(label)

    def __getitem__(self, idx):
        image = Image.open(self.imgs[idx])
        label = self.label[idx]
        image_tensor = self.transform(image)

        return image_tensor, label

    def __len__(self):
        return len(self.imgs)


class ClassificationDataModule(L.LightningDataModule):
    def __init__(self, train_path: str, test_path: str, transformers: list, batch_size: int):
        super().__init__()
        self.train_path = train_path
        self.test_path = test_path
        self.batch_size = batch_size
        self.transformers = transformers

    def setup(self, stage: str) -> None:
        if stage == "fit":
            train_data = ClassificationDataset(self.train_path, transform=self.transformers)
            self.train_subset, self.val_subset = torch.utils.data.random_split(
                train_data, [0.8, 0.2], generator=torch.Generator().manual_seed(42)
            )
        if stage == "test":
            self.test_data = ClassificationDataset(self.test_path, transform=self.transformers)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return torch.utils.data.DataLoader(
            self.train_subset, batch_size=self.batch_size, shuffle=True, num_workers=7, persistent_workers=True
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return torch.utils.data.DataLoader(
            self.val_subset, batch_size=self.batch_size, shuffle=False, num_workers=7, persistent_workers=True
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return torch.utils.data.DataLoader(
            self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=7, persistent_workers=True
        )


class Classification(L.LightningModule):
    def __init__(self, model, learning_rate, weight_decay):
        super().__init__()
        self.model = model
        self.save_hyperparameters(ignore=['model'])

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.criterion = nn.CrossEntropyLoss()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer

    def _calculate_loss_per_batch(self, data):
        images, labels = data
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)

        return loss

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        loss = self._calculate_loss_per_batch(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        val_loss = self._calculate_loss_per_batch(batch)
        self.log("val_loss", val_loss, prog_bar=True)

    def test_step(self, batch, *args: Any, **kwargs: Any):
        val_loss = self._calculate_loss_per_batch(batch)
        self.log("test_loss", val_loss, prog_bar=True)

    def predict_step(self, batch, batch_idx):
        test_images, test_labels = batch
        test_embs = self.model(test_images)
        return test_embs, test_labels


def run_model(model, model_name, data_module, dev, learning_rate):
    logger = TensorBoardLogger("../logs/", name=model_name)
    checkpoint_callback = ModelCheckpoint(dirpath=f"../logs/{model_name}/", save_top_k=1, monitor="val_loss")
    trainer = L.Trainer(accelerator=dev, devices=1, max_epochs=10, logger=logger, callbacks=[checkpoint_callback])

    learning = Classification(model=model, learning_rate=learning_rate, weight_decay=0.1)
    trainer.fit(model=learning, datamodule=data_module)

    # train
    learning = Classification.load_from_checkpoint(
        checkpoint_callback.best_model_path,  # learning_rate=learning_rate, weight_decay=0.1
    )
    # trainer.fit(model=learning, train_dataloaders=train_full_loader, val_dataloaders=test_loader)
    # test
    trainer.test(learning, datamodule=data_module)


def main():
    dev = get_device()
    print("prepare torch dataset")

    transformers = [
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
    ]
    data_module = ClassificationDataModule(
        "../Classification_data/train/", "../Classification_data/test/", transformers, 40
    )

    # KNN
    # print("KNN")
    # knn = KnnClassificate(k=3, image_size=150*150*3)
    # knn.train(train_images, train_labels)
    # predict_test_labels = knn.predict(test_images)
    # knn.evaluate_model(test_labels, predict_test_labels)

    # model CNN
    print("CNN")
    model = CNNModel(num_channels=3)
    run_model(model, "CNN", data_module, dev, 1e-4)

    # model AlexNet
    print("AlexNet")
    model_alex = AlexNet(num_classes=3)
    run_model(model_alex, "AlexNet", data_module, dev, 1e-5)

    # model with feature extractor
    print("Feature Extractor")
    train_nodes, eval_nodes = get_graph_node_names(resnet50())
    return_nodes = train_nodes[5:-2]
    model_features = CNNFeatureExtractor(return_nodes)

    run_model(model_features, "Extractor", data_module, dev, 1e-5)


if __name__ == "__main__":
    print(torch.backends.mps.is_available())
    main()
