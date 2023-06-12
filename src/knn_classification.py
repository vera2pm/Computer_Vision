from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score


class KnnClassificate:
    def __init__(self, k, image_size):
        self.knn = KNeighborsClassifier(n_neighbors=k)
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


if __name__ == "__main__":
    from src.utils import load_data
    from src.configs.classification_config import CLASS_NAMES_LABEL
    datasets_path = ["../Classification_data/train/", "../Classification_data/test/"]
    (train_images, train_labels), (test_images, test_labels) = load_data(datasets_path, CLASS_NAMES_LABEL, (150*150))

    knn = KnnClassificate(k=3, image_size=150 * 150 * 3)
    knn.train(train_images, train_labels)
    predict_test_labels = knn.predict(test_images)

    knn.evaluate_model(test_labels, predict_test_labels)
