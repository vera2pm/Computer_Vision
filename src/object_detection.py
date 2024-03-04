import torch.optim as optim
import torch
from torchmetrics.detection import MeanAveragePrecision
from torchvision.transforms import functional as F
import numpy as np


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
    def __init__(self, imgs, transforms=None):
        self.transforms = transforms
        self.imgs = imgs

    def __getitem__(self, idx):
        data = self.imgs[idx]
        image, target = data

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.imgs)


def train_detect(train_data, model, device, num_epochs=50, learning_rate=0.001, weight_decay=0.1):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model.to(device)
    train_loss_list = []
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}")
        train_loss = 0
        total = 0

        # Iterating over the training dataset in batches
        model.train()
        for i, (images, targets) in enumerate(train_data):
            # Extracting images and target labels for the batch being iterated
            # print(targets)
            images = list(image.to(device) for image in images)
            targets = [
                {key: v.to(device) if isinstance(v, torch.Tensor) else v for key, v in t.items()} for t in targets
            ]

            # Calculating the model output and the cross entropy loss
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # Updating weights according to calculated loss
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            train_loss += losses.item()
            total += len(targets)

        train_loss_list.append(train_loss / len(train_data))
        print(f"Training loss = {train_loss_list[-1]}")

    return model, train_loss_list


def eval_detect(test_data, model, device):
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
