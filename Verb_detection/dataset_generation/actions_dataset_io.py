import json
import os
import pandas as pd
import numpy as np
from os.path import isfile, join
import cv2


def json_exist(sequence_folder):
    files = [
        f
        for f in os.listdir(sequence_folder)
        if isfile(join(sequence_folder, f))
    ]
    for idx, file in enumerate(files):
        filename, file_extension = os.path.splitext(file)
        if file_extension == ".json":
            print(file, "exists, skip!")
            return True


def csv_exist(sequence_folder):
    files = [
        f
        for f in os.listdir(sequence_folder)
        if isfile(join(sequence_folder, f))
    ]
    for idx, file in enumerate(files):
        filename, file_extension = os.path.splitext(file)
        if file_extension == ".csv":
            print(file, "exists, skip!")
            return True


def read_action(dataset_dir):
    action_list = os.listdir(dataset_dir)
    return action_list


def read_sequence_index(dataset_dir, action):
    action_folder = join(dataset_dir, action)
    sequence_index_list = os.listdir(action_folder)
    return sequence_index_list


def read_image_files(dataset_dir, action, sequence):
    sequence_folder = join(dataset_dir, action, sequence)
    image_folder = join(sequence_folder, "images")
    image_files = [
        f for f in os.listdir(image_folder) if isfile(join(image_folder, f))
    ]
    return image_folder, image_files, sequence_folder


def save_hand_landmarks_image(
    sequence_index_folder, image_name, annotated_image
):
    hand_landmarks_dir = join(sequence_index_folder, "hand_landmarks_data")
    if not os.path.exists(hand_landmarks_dir):
        os.makedirs(hand_landmarks_dir)
    cv2.imwrite(os.path.join(hand_landmarks_dir, image_name), annotated_image)


def json_struct(file, result_multi_handedness, result_multi_landmarks):
    output = {
        "id": file,
        "handedness": result_multi_handedness,
        "hand_landmarks": result_multi_landmarks,
    }
    return output


def json_save(dataset_dir, action, sequence, hand_landmarks_data):
    # Write JSON file
    sequence_folder = join(dataset_dir, action, sequence)
    json_output_path = (
        sequence_folder + "/" + str(action) + "_" + sequence + ".json"
    )
    if hand_landmarks_data:
        with open(json_output_path, "w") as f:
            json.dump(hand_landmarks_data, f, indent=2)
    else:
        print("Error! No hand landmarks detected!")


def csv_save(dataset_dir, action, sequence, hand_landmarks_data):
    # Write csv file
    hand_landmarks_data = pd.concat(hand_landmarks_data)
    hand_landmarks_data = hand_landmarks_data.sort_index()
    sequence_folder = join(dataset_dir, action, sequence)
    csv_output_path = (
        sequence_folder + "/" + str(action) + "_" + sequence + ".csv"
    )
    if not hand_landmarks_data.empty:
        with open(csv_output_path, "w") as f:
            hand_landmarks_data.to_csv(f, encoding="utf-8")
            print(f"Data for action: {action}, sequence: {sequence} is saved!")
    else:
        print("Error! No hand landmarks detected!")
