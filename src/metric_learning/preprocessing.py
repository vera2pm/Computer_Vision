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
        positive_list = data_table.loc[data_table["super_class_id"] == supcl]["path"].tolist()
        positive_list.reverse()
        length = len(positive_list)
        if length % 2 != 0:
            positive_list[int((length - 1) / 2)] = positive_list[0]
        data_table.loc[data_table["super_class_id"] == supcl, "pos_list_sup"] = positive_list

        negative_list = data_table.loc[data_table["super_class_id"] != supcl]["path"].tolist()
        negative_sample = random.choices(negative_list, k=length)
        data_table.loc[data_table["super_class_id"] == supcl, "neg_list_sup"] = negative_sample

    return data_table


def cut_batch(data_table, save_path):
    index_list = []
    for supcl in data_table["super_class_id"].unique():
        index_list.extend(random.sample(data_table.loc[data_table["super_class_id"] == supcl].index.tolist(), 3))

    batch = data_table.loc[index_list]
    batch.to_csv(save_path, index=False)


def save_data(data_table, save_path):
    shape_df = preprocessing(data_table)
    # shape_df = create_pos_neg_anchor_superlabel(shape_df)
    shape_df.to_csv(save_path, index=False)


def read_data():
    train_data = pd.read_table("../../data/Stanford_Online_Products/Ebay_train.txt", sep=" ")
    test_data = pd.read_table("../../data/Stanford_Online_Products/Ebay_test.txt", sep=" ")
    # train_data = preprocessing(train_data)

    X_subtrain_inds, X_subtest_inds = next(
        GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42).split(train_data, groups=train_data["class_id"])
    )

    gss = GroupShuffleSplit(test_size=0.33, n_splits=1, random_state=42)
    X_train_inds, X_test_inds = next(gss.split(train_data, groups=train_data["class_id"]))

    X_subtrain_inds, X_subtest_inds = next(
        gss.split(train_data.iloc[X_subtest_inds], groups=train_data.iloc[X_subtest_inds]["class_id"])
    )
    print(len(X_subtrain_inds), len(X_subtest_inds), len(X_train_inds), len(X_test_inds))
    print(train_data.iloc[X_subtrain_inds]["super_class_id"].nunique())
    print(train_data.iloc[X_subtest_inds]["super_class_id"].nunique())

    cut_batch(train_data.iloc[X_train_inds], "../../data/Stanford_Online_Products/batch_train.csv")
    cut_batch(train_data.iloc[X_test_inds], "../../data/Stanford_Online_Products/batch_val.csv")
    cut_batch(test_data, "../../data/Stanford_Online_Products/batch_test.csv")

    save_data(train_data.iloc[X_subtrain_inds], "../../data/Stanford_Online_Products/Ebay_subtrain_preproc.csv")
    save_data(train_data.iloc[X_subtest_inds], "../../data/Stanford_Online_Products/Ebay_subval_preproc.csv")

    save_data(train_data.iloc[X_train_inds], "../../data/Stanford_Online_Products/Ebay_train_train_preproc.csv")
    save_data(train_data.iloc[X_test_inds], "../../data/Stanford_Online_Products/Ebay_train__val_preproc.csv")
    save_data(train_data, "../../data/Stanford_Online_Products/Ebay_train_preproc.csv")

    test_data = preprocessing(test_data)
    test_data.to_csv("../../data/Stanford_Online_Products/Ebay_test_preproc.csv", index=False)


if __name__ == "__main__":
    # print(torch.backends.mps.is_available())
    read_data()
