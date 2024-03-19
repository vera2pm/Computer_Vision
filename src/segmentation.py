import torch.optim as optim
import torch
from torch import nn
from torchmetrics.detection import MeanAveragePrecision
from torchvision.transforms import functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from dataset import BrainSegmentationDataset as Dataset
from unet import UNet
from utils import log_images, dsc


def train_val(loader_train, loader_valid, device, num_epochs=50, learning_rate=0.001, weight_decay=0.1):
    loaders = {"train": loader_train, "valid": loader_valid}

    unet = UNet(in_channels=Dataset.in_channels, out_channels=Dataset.out_channels)
    unet.to(device)

    dsc_loss = nn.CrossEntropyLoss()
    best_validation_dsc = 0.0

    optimizer = optim.Adam(unet.parameters(), lr=learning_rate)

    # logger = logging
    loss_train = []
    loss_valid = []

    step = 0

    for epoch in tqdm(range(num_epochs), total=num_epochs):
        for phase in ["train", "valid"]:
            if phase == "train":
                unet.train()
            else:
                unet.eval()

            validation_pred = []
            validation_true = []

            for i, data in enumerate(loaders[phase]):
                if phase == "train":
                    step += 1

                x, y_true = data
                x, y_true = x.to(device), y_true.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    y_pred = unet(x)

                    loss = dsc_loss(y_pred, y_true)

                    if phase == "valid":
                        loss_valid.append(loss.item())
                        y_pred_np = y_pred.detach().cpu().numpy()
                        validation_pred.extend([y_pred_np[s] for s in range(y_pred_np.shape[0])])
                        y_true_np = y_true.detach().cpu().numpy()
                        validation_true.extend([y_true_np[s] for s in range(y_true_np.shape[0])])
                        # if (epoch % args.vis_freq == 0) or (epoch == num_epochs - 1):
                        # if i * args.batch_size < args.vis_images:
                        #     tag = "image/{}".format(i)
                        #     num_images = args.vis_images - i * args.batch_size
                        #     logger.image_list_summary(
                        #         tag,
                        #         log_images(x, y_true, y_pred)[:num_images],
                        #         step,
                        #     )

                    if phase == "train":
                        loss_train.append(loss.item())
                        loss.backward()
                        optimizer.step()

                if phase == "train" and (step + 1) % 10 == 0:
                    # log_loss_summary(logger, loss_train, step)
                    loss_train = []

            if phase == "valid":
                # log_loss_summary(logger, loss_valid, step, prefix="val_")
                mean_dsc = np.mean(
                    dsc_per_volume(
                        validation_pred,
                        validation_true,
                        loader_valid.dataset.patient_slice_index,
                    )
                )
                # logger.scalar_summary("val_dsc", mean_dsc, step)
                if mean_dsc > best_validation_dsc:
                    best_validation_dsc = mean_dsc
                    torch.save(unet.state_dict(), os.path.join(args.weights, "unet.pt"))
                loss_valid = []

    print("Best validation mean DSC: {:4f}".format(best_validation_dsc))
    return train_val_loss_list, test_acc_list


def eval_segmentation(test_data, model, device):
    model.to(device)
    test_loss_list = []
    model.eval()
    metric = MeanAveragePrecision(iou_type="bbox")
    for i, (images, targets) in enumerate(test_data):
        images = list(img.to(device) for img in images)

        predictions = model(images)
        predictions = [{k: v.to(device) for k, v in t.items()} for t in predictions]

        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        print(targets)

        # gather the stats from all processes
        test_loss_list.append(metric(predictions, targets))

    # torch.set_num_threads(n_threads)
    return predictions, metric, test_loss_list


if __name__ == "__main__":
    print(torch.backends.mps.is_available())
