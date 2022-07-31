import model.data_preprocess
import model.params
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split


def get_data():
    X, y = get_all_data()
    data_X, data_y = data_preprocess.data_window(X, y)
    X_train, X_test, y_train, y_test = train_test_split(
        data_X, data_y, test_size=0.3, random_state=42
    )
    X_train = np.asarray(X_train).astype("float32")
    X_test = np.asarray(X_test).astype("float32")
    y_train = np.asarray(y_train).astype("float32").reshape((-1, 12))
    y_test = np.asarray(y_test).astype("float32").reshape((-1, 12))

    return X_train, X_test, y_train, y_test


def get_all_data():
    # get data, slide it into 3D, return training and testing sets
    record_list = os.listdir(params.dataset_dir)
    X = []
    y = []
    for record in record_list:
        X_record, y_record = get_record_data(record)
        n_stride = len(X_record) // params.stride
        length = n_stride * params.stride
        X_record = X_record.tail(length)
        y_record = y_record.tail(length)

        X.append(X_record)
        y.append(y_record)

    X = pd.concat(X)
    y = pd.concat(y)

    return X, y


def get_record_data(record):
    data_raw = get_raw_data(record)
    data = data_raw.copy()

    X = data.drop(columns=params.classes.keys())
    X = X.drop(columns="timestamp")
    X = X.iloc[:, 1:]
    y = data[params.classes.keys()]

    return X, y


def get_raw_data(record):
    # get raw data
    if not data_preprocess.csv_exist(params.dataset_dir + "/" + record):
        data = data_preprocess.append_by_sequence(params.get_path(record))
    else:
        path = params.dataset_dir + "/" + record + "/data_int.csv"
        data = pd.read_csv(path)
    return data


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_data()
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
