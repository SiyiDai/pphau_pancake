from dataset_generation.hand_landmarks_detector import mp_process, get_hand
import pandas as pd
import cv2
import mediapipe as mp
import numpy as np

from Verb_detection.model import params
import os
from os.path import isfile, join
import time


# Input: frames
# Output: verb

model_name = "finalized_model_5_bags_LSTM_128.sav"
VISUALIZE = False

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
# model = pickle.load(open(model_name, "rb"))


def read_image_files(dir):
    image_files = [f for f in os.listdir(dir) if isfile(join(dir, f))]
    return image_files


def process_verb(image_batch, model, labels):
    # remeber to check whether the model is correcly saved!!
    if check_batches_size(image_batch):
        data = process_mediapipe(image_batch)
        # print("MediaPipe data shape:", data.shape)
        t, pred = model_predict(data, model)

        verb = get_key_from_val(labels, pred)
        return t, verb


def data_window(X):
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

    return data_X


def process_mediapipe(image_batch):
    # The input should be a batch of [timestamps, images], size = model.params.time_steps
    hand_landmarks_data = []
    for [timestamp, image] in image_batch:
        output, annotated_image = detect_landmarks(image, timestamp)
        hand_landmarks_data.append(output)
    if hand_landmarks_data:
        hand_landmarks_data = pd.concat(hand_landmarks_data)
        # sort by timestamp
        hand_landmarks_data = hand_landmarks_data.sort_index(0)
        hand_landmarks_data = check_timestamp(hand_landmarks_data)
        # Drop the timestamp columns of dataframe
        data_appended_all = hand_landmarks_data.iloc[:, 1:]
        data_appended_all = data_window(data_appended_all)
        data_appended_all = np.asarray(data_appended_all).astype("float32")
    return data_appended_all


def df_to_add(new_ts):
    # create zero paddings for non-detection images
    df_add = {
        "timestamp": new_ts,
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
    # check if all timestamp is continuous
    ts = df["timestamp"].to_list()
    for idx, ts_current in enumerate(ts):
        gap = ts_current - ts[idx - 1]
        if gap != 1:
            for i in range(1, gap):
                df_add = df_to_add(int(ts[idx - 1] + i))
                df = df.append(df_add, ignore_index=True)
    df = df.sort_values(by="timestamp")
    return df


def get_key_from_val(dic, val):
    keys = [k for k, v in dic.items() if v == val]
    return keys[0]


def model_predict(data, model):
    start = time.time()
    data_pred = model.predict_on_batch(data)
    end = time.time()
    pred = np.array(np.argmax(data_pred, axis=1))
    t = end - start
    return t, pred


def check_batches_size(image_batch):
    if len(image_batch) == params.time_steps:
        return True
    else:
        print(
            len(image_batch),
            "is not fit for required image batch length for current model:",
            params.time_steps,
        )


def detect_landmarks(image, timestamp):
    results = mp_process(image)
    if results.multi_hand_landmarks:
        result_multi_landmarks = format_results_to_dataframe(results, timestamp)

        # if VISUALIZE:
        annotated_image = visualize_hand_landmarks(image, results)
    else:
        zero_padded = df_to_add(timestamp)
        result_multi_landmarks = pd.DataFrame(zero_padded, index=[timestamp])
        annotated_image = image.copy()
    return result_multi_landmarks, annotated_image


def format_results_to_dataframe(results, timestamp):
    data = {"timestamp": timestamp}
    for idx_hl, hand_landmarks in enumerate(results.multi_hand_landmarks):
        classification = list(results.multi_handedness[idx_hl].classification)
        # 0: Left, 1: Right
        hand = get_hand(classification)
        for idx_dp, data_point in enumerate(hand_landmarks.landmark):
            data[f"{hand}_{idx_dp}_x"] = data_point.x
            data[f"{hand}_{idx_dp}_y"] = data_point.y
            data[f"{hand}_{idx_dp}_z"] = data_point.z
    # check how many hands were detected, if only one hand was detected, fill the other as 0
    if len(data) == 21 * 3 + 1:
        idx_hl_fill = 1 - int(hand)
        for idx_dp in range(21):
            data[f"{idx_hl_fill}_{idx_dp}_x"] = 0.0
            data[f"{idx_hl_fill}_{idx_dp}_y"] = 0.0
            data[f"{idx_hl_fill}_{idx_dp}_z"] = 0.0

    data_pd = pd.DataFrame(data, index=[timestamp])
    return data_pd


def visualize_hand_landmarks(
    image,
    results,
    # debug=False,
):
    image_height, image_width, _ = image.shape
    # visualize the hand landmarks to image
    annotated_image = image.copy()
    for hand_landmarks in results.multi_hand_landmarks:
        # if debug:
        #     print("hand_landmarks:", hand_landmarks)
        #     print(
        #         f"Index finger tip coordinates: (",
        #         f"{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, "
        #         f"{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})",
        #     )
        mp_drawing.draw_landmarks(
            annotated_image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            # mp_drawing_styles.get_default_hand_landmarks_style(),
            # mp_drawing_styles.get_default_hand_connections_style(),
        )
    return annotated_image
    # Debug
    # while debug == True:
    #     cv2.imshow("mediapipe_hand_image", annotated_image)
    #     key = cv2.waitKey(1)
    #     # if pressed escape exit program
    #     if key == 27:
    #         cv2.destroyAllWindows()
    #         break


# if __name__ == "__main__":
#     dir = "test"
#     images = read_image_files(dir)
#     image_batch = []
#     for idx, file in enumerate(images):
#         image = cv2.imread(os.path.join(dir, file))
#         image_batch.append([idx, image])
#     t_long = []
#     for i in range(100):
#         t, verb = process_verb(image_batch, model)
#         t_long.append(t)
#     t_long.pop(0)
#     print("The current action is:", verb)
#     print("The average inference time is:", np.mean(t_long) * 1000)
#     print("The standard deviation of inference time is:", np.std(t_long) * 1000)
