"""
Project: Hand sign emoji Recognition for Video Calling using CNN
@Author: Pravesh Bawangade
"""
import os
import pandas as pd


def create_csv(data_path="Data", per=0.7) -> bool:
    """
    Create a CSV file with paths of image and their corresponding labels.
    :param data_path: Path where data is present
    :param per: Percentage for train/test split.
    :return: bool
    """
    data = {"path": [], "labels": []}
    if not os.path.exists(data_path):
        print("Path doesn't exists.")
        return False

    for folder in os.listdir(data_path):
        if folder.startswith("."):
            pass
        else:
            data["path"].extend([data_path + "/" + folder + "/" + img for img in os.listdir(data_path + "/" + folder)])
            data["labels"].extend([folder for i in range(len(os.listdir(data_path + "/" + folder)))])

    df_data = pd.DataFrame(data)
    df_data = df_data.sample(frac=1).reset_index(drop=True)
    num_train = int(df_data.shape[0] * per)
    train = df_data.iloc[0:num_train, :]
    test = df_data.iloc[num_train:, :]
    train.to_csv("train.csv")
    test.to_csv("test.csv")

    return True


if __name__ == "__main__":
    ret = create_csv(data_path="Data", per=0.7)
    print(ret)
