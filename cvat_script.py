import logging, os
from cvat_sdk import make_client
import torchvision.transforms as transforms
from cvat_sdk.pytorch import TaskVisionDataset, ExtractBoundingBoxes
import torch

logging.basicConfig(level=logging.INFO,
      format='%(levelname)s - %(message)s')

client = make_client('http://localhost:8080/', credentials=("Vera_Kochetkova", 'vUS7z4!8.JXfELW'))

TRAIN_TASK_ID = 1
for task_id in [1, 2]:
    train_set = TaskVisionDataset(client, task_id,
                                  transform=transforms.ToTensor(),
                                  target_transform=ExtractBoundingBoxes(include_shape_types=['polyline'])
                                  )
    print(len(train_set))
    # print(train_set[0])
    print(type(train_set[0]))
    torch.save([train_tup for train_tup in train_set], f"train_set_{task_id}.pt")

