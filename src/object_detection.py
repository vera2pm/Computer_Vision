import torch.optim as optim
import torch


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

            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

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


