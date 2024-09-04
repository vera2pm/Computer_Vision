import random

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


def create_pos_neg_anchor_superlabel(data_table):
    for supcl in data_table["super_class_id"].unique():
        positive_list = data_table.loc[data_table["super_class_id"] == supcl]["path"].tolist().reverse()
        length = len(positive_list)
        if length % 2 != 0:
            positive_list[int((length - 1) / 2)] = positive_list[0]
        data_table.loc[data_table["super_class_id"] == supcl, "pos_list_sup"] = positive_list

        negative_list = data_table.loc[data_table["super_class_id"] != supcl]["path"].tolist()
        negative_sample = random.sample(negative_list, length)
        data_table.loc[data_table["super_class_id"] == supcl, "neg_list_sup"] = negative_sample

    return data_table


def save_data(data_table, save_path):
    shape_df = preprocessing(data_table)
    shape_df = create_pos_neg_anchor_superlabel(shape_df)
    shape_df.to_csv(save_path)


def read_data():
    train_data = pd.read_table("../../data/Stanford_Online_Products/Ebay_train.txt", sep=" ")
    test_data = pd.read_table("../../data/Stanford_Online_Products/Ebay_test.txt", sep=" ")
    shape_df = preprocessing(train_data)

    gss = GroupShuffleSplit(test_size=0.33, n_splits=2, random_state=42).split(shape_df, groups=shape_df["class_id"])

    X_subtrain_inds, X_subtest_inds, X_train_inds, X_test_inds = next(gss)
    print(len(X_subtrain_inds), len(X_subtest_inds), len(X_train_inds), len(X_test_inds))
    shape_df.iloc[X_train_inds].to_csv("../../data/Stanford_Online_Products/Ebay_train_train_preproc.csv", index=False)
    shape_df.iloc[X_test_inds].to_csv("../../data/Stanford_Online_Products/Ebay_train__val_preproc.csv", index=False)
    shape_df.to_csv("../../data/Stanford_Online_Products/Ebay_train_preproc.csv", index=False)

    shape_df = preprocessing(test_data)
    shape_df.to_csv("../../data/Stanford_Online_Products/Ebay_test_preproc.csv", index=False)


if __name__ == "__main__":
    # print(torch.backends.mps.is_available())
    read_data()
