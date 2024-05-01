dataset_path = "../data/segmentation_dataset/"

train_path = dataset_path + "train_data/"
train_labels_file = train_path + "train_labels.json"
train_files = train_path + "target_amount.json"

test_path = dataset_path + "test_data/"
test_labels_file = test_path + "test_labels.json"
test_files = test_path + "target_amount.json"


IN_CHANNELS = 3
OUT_CHANNELS = 1

IMAGE_SIZE = (256, 256)
