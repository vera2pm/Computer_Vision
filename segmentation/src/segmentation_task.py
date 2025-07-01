import matplotlib.pyplot as plt
import seaborn as sns
import torch
import albumentations as A
import sys

sys.path.append("../")
from src.segmentation.segmentation import SegmentationDataset, Segmentation, SegmentationDataModule
from src.configs.segment_config import IMAGE_SIZE, train_path, test_path, train_files, test_files
from src.segmentation.unet_model import UNet

from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import lightning as L

from src.utils import get_device


def plot_train_val(df, title="Segmentation UNET model"):
    fig = plt.figure(figsize=(15, 5))
    ax0 = fig.add_subplot(121, title="Train Loss", xlabel="Epoch")
    # ax1 = fig.add_subplot(122, title="Validation Loss", xlabel="Epoch")
    sns.lineplot(data=df.set_index("epoch"), ax=ax0)
    fig.suptitle(title)
    plt.savefig(f"../{title}.jpg")


def main():
    # prepare dataset
    print("prepare dataset")
    # prepare torch dataset
    transformers = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.GaussianBlur(3),
    ]
    data_module = SegmentationDataModule(train_path, test_path, train_files, test_files, transformers, 15)

    model = UNet(in_channels=SegmentationDataset.in_channels, out_channels=SegmentationDataset.out_channels)

    model_name = "UNet_aug"
    dev = get_device()

    logger = TensorBoardLogger("../logs/", name=model_name)
    checkpoint_callback = ModelCheckpoint(dirpath=f"../logs/{model_name}/", save_top_k=1, monitor="val_loss")
    trainer = L.Trainer(
        accelerator=dev, devices=1, max_epochs=300, logger=logger, callbacks=[checkpoint_callback], precision="16-mixed"
    )

    learning = Segmentation(model, learning_rate=1.0e-5, weight_decay=0.1)
    trainer.fit(model=learning, datamodule=data_module)

    learning = Segmentation.load_from_checkpoint(checkpoint_callback.best_model_path)
    trainer.test(learning, datamodule=data_module)


if __name__ == "__main__":
    print(torch.backends.mps.is_available())
    main()
