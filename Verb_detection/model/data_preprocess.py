import params
import pandas as pd
import os
import numpy as np


def write_labels_to_columns(data):
    for key in params.classes.keys():
        if not key in data.columns:
            data[key] = 0
    ts = data["timestamp"].to_list()
    for idx, ts_current in enumerate(ts):
        index = data[data["timestamp"] == ts_current].index.values[0]
        class_name = data["action"][index]
        # print(class_num)
        # class_key = params.classes[class_name]
        data.at[index, class_name] = 1
        # print(data_new.loc[index].at[class_name])
    return data


def data_window(X, y):
    # make a sliding window with stride and time_steps
    # turn the originally 2D dataframe into 3D for LSTM
    # this will add into the list as [[0,1...n],[10,11...n+10],...[m-10,m-9...m]]

    sequences = params.time_steps
    result = []
    n_stride = len(X) // params.stride
    for j in range(n_stride - 1):
        start = j * params.stride
        if start + sequences <= len(X):

            result.append(X[start : start + sequences])

    data_X = np.array(result, dtype=object)
    # making the y into same length as X
    data_y = np.array(y.tail(data_X.shape[0]).values)

    return data_X, data_y


def append_by_sequence(path):
    # append sequences together
    # folder format:  action/sequences
    data_appended_all = []
    for p in path:
        data = append_by_record(p)
        save_data_csv(p, data)
        data_appended_all.append(data)

    data_appended_all = pd.concat(data_appended_all)
    data_appended_all = data_appended_all.sort_values(by="timestamp")
    data_appended_all = check_timestamp(data_appended_all)
    # Drop the original index columns of dataframe
    data_appended_all = data_appended_all.iloc[:, 1:]

    save_data_csv(os.path.dirname(path[0]), data_appended_all)
    data = save_data_int_csv(os.path.dirname(path[0]), data_appended_all)

    return data


def save_data_int_csv(path, data):
    # data with object label as int
    csv_output_path_test = path + "/data_int.csv"
    data = write_labels_to_columns(data)
    data = data.drop(columns=["action"])
    data.to_csv(csv_output_path_test, encoding="utf-8")
    return data


def save_data_csv(path, data):
    # data with object label as string
    csv_output_path = path + "/data.csv"
    data.to_csv(csv_output_path, encoding="utf-8")


def rewrite_label_to_int(data):
    # write action label to int
    class_map = params.classes
    data = data.applymap(lambda s: class_map.get(s) if s in class_map else s)
    return data


def csv_exist(path):
    # check if data_all is already exported
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    for idx, file in enumerate(files):
        filename, file_extension = os.path.splitext(file)
        if file_extension == ".csv":
            print(file, "already generated, skip!")
            return True


def append_by_record(path):
    # append .csv by record
    sequence_list = os.listdir(path)
    action = path.split("/")[-1]
    data_appended = []
    for _, sequence in enumerate(sequence_list):
        csv_name = action + "_" + sequence + ".csv"
        data = pd.read_csv(os.path.join(path, sequence, csv_name))
        data_appended.append(data)
    data_appended = pd.concat(data_appended)
    return data_appended


def df_to_add(new_ts):
    # create "other" label actions for non-detections and free-motions
    df_add = {
        "Unnamed: 0": new_ts,
        "timestamp": new_ts,
        "action": "other",
        "0_0_x": 0,
        "0_0_y": 0,
        "0_0_z": 0,
        "0_1_x": 0,
        "0_1_y": 0,
        "0_1_z": 0,
        "0_2_x": 0,
        "0_2_y": 0,
        "0_2_z": 0,
        "0_3_x": 0,
        "0_3_y": 0,
        "0_3_z": 0,
        "0_4_x": 0,
        "0_4_y": 0,
        "0_4_z": 0,
        "0_5_x": 0,
        "0_5_y": 0,
        "0_5_z": 0,
        "0_6_x": 0,
        "0_6_y": 0,
        "0_6_z": 0,
        "0_7_x": 0,
        "0_7_y": 0,
        "0_7_z": 0,
        "0_8_x": 0,
        "0_8_y": 0,
        "0_8_z": 0,
        "0_9_x": 0,
        "0_9_y": 0,
        "0_9_z": 0,
        "0_10_x": 0,
        "0_10_y": 0,
        "0_10_z": 0,
        "0_11_x": 0,
        "0_11_y": 0,
        "0_11_z": 0,
        "0_12_x": 0,
        "0_12_y": 0,
        "0_12_z": 0,
        "0_13_x": 0,
        "0_13_y": 0,
        "0_13_z": 0,
        "0_14_x": 0,
        "0_14_y": 0,
        "0_14_z": 0,
        "0_15_x": 0,
        "0_15_y": 0,
        "0_15_z": 0,
        "0_16_x": 0,
        "0_16_y": 0,
        "0_16_z": 0,
        "0_17_x": 0,
        "0_17_y": 0,
        "0_17_z": 0,
        "0_18_x": 0,
        "0_18_y": 0,
        "0_18_z": 0,
        "0_19_x": 0,
        "0_19_y": 0,
        "0_19_z": 0,
        "0_20_x": 0,
        "0_20_y": 0,
        "0_20_z": 0,
        "1_0_x": 0,
        "1_0_y": 0,
        "1_0_z": 0,
        "1_1_x": 0,
        "1_1_y": 0,
        "1_1_z": 0,
        "1_2_x": 0,
        "1_2_y": 0,
        "1_2_z": 0,
        "1_3_x": 0,
        "1_3_y": 0,
        "1_3_z": 0,
        "1_4_x": 0,
        "1_4_y": 0,
        "1_4_z": 0,
        "1_5_x": 0,
        "1_5_y": 0,
        "1_5_z": 0,
        "1_6_x": 0,
        "1_6_y": 0,
        "1_6_z": 0,
        "1_7_x": 0,
        "1_7_y": 0,
        "1_7_z": 0,
        "1_8_x": 0,
        "1_8_y": 0,
        "1_8_z": 0,
        "1_9_x": 0,
        "1_9_y": 0,
        "1_9_z": 0,
        "1_10_x": 0,
        "1_10_y": 0,
        "1_10_z": 0,
        "1_11_x": 0,
        "1_11_y": 0,
        "1_11_z": 0,
        "1_12_x": 0,
        "1_12_y": 0,
        "1_12_z": 0,
        "1_13_x": 0,
        "1_13_y": 0,
        "1_13_z": 0,
        "1_14_x": 0,
        "1_14_y": 0,
        "1_14_z": 0,
        "1_15_x": 0,
        "1_15_y": 0,
        "1_15_z": 0,
        "1_16_x": 0,
        "1_16_y": 0,
        "1_16_z": 0,
        "1_17_x": 0,
        "1_17_y": 0,
        "1_17_z": 0,
        "1_18_x": 0,
        "1_18_y": 0,
        "1_18_z": 0,
        "1_19_x": 0,
        "1_19_y": 0,
        "1_19_z": 0,
        "1_20_x": 0,
        "1_20_y": 0,
        "1_20_z": 0,
    }
    return df_add


def check_timestamp(df):
    # check if all timestamp is continuous, otherwise label it as "other"
    ts = df["timestamp"].to_list()
    for idx, ts_current in enumerate(ts):
        gap = ts_current - ts[idx - 1]
        if gap != 1:
            for i in range(1, gap):
                df_add = df_to_add(int(ts[idx - 1] + i))
                df = df.append(df_add, ignore_index=True)
    df = df.sort_values(by="timestamp")
    return df


def get_actual_pos():
    """
    get actual z from depth information
    input: x, y in landmarks
    output: x, y, z in real pos
    """
    pass


def data_augmentation():
    """
    add some random noise for augmentation
    """
    pass
