from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score


class KnnClassificate:
    def __init__(self, k, n_images, image_size):
        self.knn = KNeighborsClassifier(n_neighbors=k)
        self.n_images = n_images
        self.image_size = image_size

    def preprocess(self, image_array):
        image_array_reshape = np.array(image_array).reshape((len(image_array), self.image_size))
        return image_array_reshape

    def train(self, train_images, train_labels):
        train_images_reshape = self.preprocess(train_images)
        self.knn.fit(train_images_reshape, train_labels)

    def predict(self, test_images):
        test_images_reshape = self.preprocess(test_images)
        predicted_labels = self.knn.predict(test_images_reshape)
        return predicted_labels

    def evaluate_model(self, true_labels, predicted_labels):
        accur = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels, average="macro")
        recall = recall_score(true_labels, predicted_labels, average="macro")
        print(f"Accuracy {accur}")
        print(f"Precision {precision}")
        print(f"Recall {recall}")

