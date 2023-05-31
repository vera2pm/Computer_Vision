import tqdm
import cv2
import numpy as np


def preprocess(images_dataset: list, image_size):
    images = []
    for image in tqdm(images_dataset):
        image = cv2.resize(image, image_size)
        images.append(image)
    images = np.array(images, dtype='float32')

    return images
