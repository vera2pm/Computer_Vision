import os
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score
from tqdm import tqdm

from src.configs.classification_config import CLASS_NAMES_LABEL, IMAGE_SIZE
from src.utils import cv2_load2rgb


class KnnClassificate:
    def __init__(self, train_path, test_path, image_size, k):
        self.knn = KNeighborsClassifier(n_neighbors=k)
        # self.data_module = data_module
        self.image_size = image_size
        self.train_data = self.load_data(train_path)
        self.test_data = self.load_data(test_path)

    def load_data(self, data_path, class_names_label=CLASS_NAMES_LABEL):
        imgs = []
        labels = []
        for folder in os.listdir(data_path):
            if folder not in class_names_label.keys():
                continue
            label = class_names_label[folder]
            for file in tqdm(os.listdir(os.path.join(data_path, folder))):
                # Get the path name of the image
                image = cv2_load2rgb(os.path.join(os.path.join(data_path, folder), file))
                image = cv2.resize(image, IMAGE_SIZE)
                imgs.append(image)
                labels.append(label)

        return imgs, labels

    def preprocess(self, image_array):
        image_array_reshape = np.array(image_array).reshape((len(image_array), self.image_size))
        return image_array_reshape

    def train(self):
        train_images, train_labels = self.train_data
        train_images_sh = self.preprocess(train_images)

        self.knn.fit(train_images_sh, train_labels)

    def predict(self):
        test_images, true_labels = self.test_data

        test_images = self.preprocess(test_images)
        predicted_labels = self.knn.predict(test_images)

        self.evaluate_model(true_labels, predicted_labels)
        return predicted_labels

    def evaluate_model(self, true_labels, predicted_labels):
        accur = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels, average="macro")
        recall = recall_score(true_labels, predicted_labels, average="macro")
        print(f"Accuracy {accur}")
        print(f"Precision {precision}")
        print(f"Recall {recall}")
