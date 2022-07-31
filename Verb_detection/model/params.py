import os

MODEL_PATH = os.path.dirname(__file__)
ROOT_PATH = os.path.dirname(MODEL_PATH)
dataset_dir = os.path.join(ROOT_PATH, "actions_dataset")

stride = 1
time_steps = 10
n_features = 126  # 21x3x2=126
# 21 points x (x, y, z) x (left, right)


# sample_weight = {
#     0: 15.63 / 84,
#     1: 15.63 / 330,
#     2: 15.63 / 101,
#     3: 15.63 / 100,
#     4: 15.63 / 110,
#     5: 15.63 / 560,
#     6: 15.63 / 214,
#     7: 15.63 / 311,
#     8: 15.63 / 413,
#     9: 15.63 / 331,
#     10: 15.63 / 209,
#     11: 15.63 / 6539,
# }

params_model = {
    "0_LSTM": 128,
    "0_input_shape": (time_steps, n_features),
    "1_dropout": 0.2,
    # "2_LSTM": 64,
    # "3_LSTM": 64,
    # "4_LSTM": 128,
    # "5_dropout": 0.2,
    # "6_time_distributed": n_features,
    "2_flatten": "flatten",
    "3_dense": 128,
    "4_dense": 12,
}

params_fit = {
    "patience": 8,
    "validation_split": 0.3,
    "epochs": 100,
    # "sample_weight": sample_weight,
    "batch_size": 32,
    "verbose": 0,
}

classes = {
    "close": 0,
    "decorate": 1,
    "flip": 2,
    "move_object": 3,
    "open": 4,
    "pick_up": 5,
    "pour": 6,
    "put_down": 7,
    "screw": 8,
    "shovel": 9,
    "squeeze": 10,
    "other": 11,
}


def get_path(record):
    path_0 = dataset_dir + "/" + record + "/close"
    path_1 = dataset_dir + "/" + record + "/decorate"
    path_2 = dataset_dir + "/" + record + "/flip"
    path_3 = dataset_dir + "/" + record + "/move_object"
    path_4 = dataset_dir + "/" + record + "/open"
    path_5 = dataset_dir + "/" + record + "/pick_up"
    path_6 = dataset_dir + "/" + record + "/pour"
    path_7 = dataset_dir + "/" + record + "/put_down"
    path_8 = dataset_dir + "/" + record + "/screw"
    path_9 = dataset_dir + "/" + record + "/shovel"
    path_10 = dataset_dir + "/" + record + "/squeeze"
    path_11 = dataset_dir + "/" + record + "/other"

    path = [
        path_0,
        path_1,
        path_2,
        path_3,
        path_4,
        path_5,
        path_6,
        path_7,
        path_8,
        path_9,
        path_10,
        path_11,
    ]

    return path
