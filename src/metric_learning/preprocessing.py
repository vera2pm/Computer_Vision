from PIL import Image
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def data_cleaning():
    pass


def preprocessing(data):
    # images = data["path"].values
    shape_df = []
    for i, row in data.iterrows():
        image = Image.open("../../data/Stanford_Online_Products/" + row["path"])
        (w, h), n = image.size, image.mode
        if n != "RGB":
            # print(row["path"])
            continue
        shape_df.append([row["path"], row["class_id"], row["super_class_id"], n])
    shape_df = pd.DataFrame(shape_df, columns=["path", "class_id", "super_class_id", "n"])
    class_count = shape_df.groupby("class_id")["path"].count()
    class_count.name = "count"
    shape_df = shape_df.merge(class_count.reset_index(), on="class_id", how="left")
    shape_df = shape_df.loc[shape_df["count"] != 1]
    return shape_df


def read_data():
    train_data = pd.read_table("../../data/Stanford_Online_Products/Ebay_train.txt", sep=" ")
    test_data = pd.read_table("../../data/Stanford_Online_Products/Ebay_test.txt", sep=" ")
    shape_df = preprocessing(train_data)

    gss = GroupShuffleSplit(test_size=0.33, n_splits=1, random_state=42).split(
        shape_df, groups=shape_df["super_class_id"]
    )

    X_train_inds, X_test_inds = next(gss)
    shape_df.iloc[X_train_inds].to_csv("../../data/Stanford_Online_Products/Ebay_train_train_preproc.csv", index=False)
    shape_df.iloc[X_test_inds].to_csv("../../data/Stanford_Online_Products/Ebay_train__val_preproc.csv", index=False)
    shape_df.to_csv("../../data/Stanford_Online_Products/Ebay_train_preproc.csv", index=False)

    shape_df = preprocessing(test_data)
    shape_df.to_csv("../../data/Stanford_Online_Products/Ebay_test_preproc.csv", index=False)


if __name__ == "__main__":
    # print(torch.backends.mps.is_available())
    read_data()
