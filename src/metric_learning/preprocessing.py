from PIL import Image
import pandas as pd


def data_cleaning():
    pass


def preprocessing(data):
    # images = data["path"].values
    shape_df = []
    for i, row in data.iterrows():
        image = Image.open("../../data/Stanford_Online_Products/" + row["path"])
        (w, h), n = image.size, image.mode
        if n != "RGB":
            print(row["path"])
            continue
        shape_df.append([row["path"], row["class_id"], row["super_class_id"], n])
    shape_df = pd.DataFrame(shape_df, columns=["path", "class_id", "super_class_id", "n"])
    return shape_df


def read_data():
    train_data = pd.read_table("../../data/Stanford_Online_Products/Ebay_train.txt", sep=" ")
    test_data = pd.read_table("../../data/Stanford_Online_Products/Ebay_test.txt", sep=" ")
    shape_df = preprocessing(train_data)
    shape_df.to_csv("../../data/Stanford_Online_Products/Ebay_train_preproc.csv", index=False)

    shape_df = preprocessing(test_data)
    shape_df.to_csv("../../data/Stanford_Online_Products/Ebay_test_preproc.csv", index=False)


if __name__ == "__main__":
    # print(torch.backends.mps.is_available())
    read_data()
