import json
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import cKDTree

from src.configs.segment_config import train_path, test_path, train_labels_file, test_labels_file
from src.utils import cv2_load2rgb


def get_mask(image, regions):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_mask = np.zeros(image.shape, dtype=np.uint8)
    coords_list = []
    if len(regions) == 0:
        return 255 - image_mask
    elif len(regions) == 1:
        radius = 10
    else:
        # radius is min distance between points divided on 2
        for reg in regions.keys():
            reg = regions[reg]["shape_attributes"]
            coords_list.append([reg["cx"], reg["cy"]])
        coords_list = np.array(coords_list)
        min_dists, min_dist_idx = cKDTree(coords_list).query(coords_list, 2)
        min_dist = min([dist[1] for dist in min_dists])
        radius = np.int32(min_dist / 2)
    for reg in regions.keys():
        reg = regions[reg]["shape_attributes"]
        center = (int(reg["cx"]), int(reg["cy"]))
        image_mask = cv2.circle(image_mask, center, radius, color=(255, 255, 255), thickness=-1)
    print(image_mask.dtype)
    # img2_fg = cv2.bitwise_and(image, image, mask=image_mask)
    # plt.imshow(image_mask)
    # plt.show()
    return image_mask


def create_targets(images_dict_file, img_dir):
    with open(images_dict_file) as f:
        labels_dict = json.load(f)

    loaded_target_amount = {}
    for idx, label_dict in labels_dict.items():
        img_filename = os.path.join(img_dir, label_dict["filename"])
        print(img_filename)
        image = cv2_load2rgb(img_filename)
        # loaded_images.append(Image.open(img_filename))
        target_regions = label_dict["regions"]
        loaded_target_amount[label_dict["filename"]] = len(target_regions)
        image_mask = get_mask(image, target_regions)
        mask_path = os.path.join(img_dir, f'mask_{label_dict["filename"]}')
        # break
        cv2.imwrite(mask_path, image_mask)

    print(loaded_target_amount)

    with open(os.path.join(img_dir, "target_amount.json"), "w") as f:
        json.dump(loaded_target_amount, f)


def save_loss_summary(loss_list):
    max(loss_list)


def check_size(amount_dict_file, img_dir):
    with open(amount_dict_file) as f:
        amount_dict = json.load(f)
    widths = []
    heights = []
    for img_name, amount in amount_dict.items():
        img_filename = os.path.join(img_dir, img_name)
        img = cv2_load2rgb(img_filename)
        print(img.shape)
        widths.append(img.shape[0])
        heights.append(img.shape[1])
    print(f"mean sizes: {np.mean(widths)}, {np.mean(heights)}")
    print(f"min sizes: {np.min(widths)}, {np.min(heights)}")
    print(f"max sizes: {np.max(widths)}, {np.max(heights)}")


if __name__ == "__main__":
    create_targets(test_labels_file, test_path)
    create_targets(train_labels_file, train_path)
    # check_size(train_files, "../../data/segmentation_dataset/train_data/")
