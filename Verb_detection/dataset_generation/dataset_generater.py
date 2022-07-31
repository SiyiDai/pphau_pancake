import os
from hand_landmarks_detector import detect_landmarks
from actions_dataset_io import (
    csv_exist,
    csv_save,
    read_action,
    read_sequence_index,
    read_image_files,
)

ROOT_PATH = os.path.dirname(__file__)
DATASET_DIR = os.path.join(ROOT_PATH, "../actions_dataset")


def dataset_generater():
    dataset_dir = DATASET_DIR
    action_list = read_action(dataset_dir)

    for _, action in enumerate(action_list):
        sequence_index_list = read_sequence_index(dataset_dir, action)

        for _, sequence in enumerate(sequence_index_list):
            image_folder, image_files, sequence_folder = read_image_files(
                dataset_dir, action, sequence
            )

            # if json_exist(join(dataset_dir, action, sequence)):
            if csv_exist(os.path.join(dataset_dir, action, sequence)):
                # if json already saved, continue
                continue

            hand_landmarks_data = []
            for _, file in enumerate(image_files):
                # output = detect_results(image_folder, file,action)
                output = detect_landmarks(
                    image_folder, file, action, sequence_folder
                )
                hand_landmarks_data.append(output)
            if hand_landmarks_data:
                # json_save(dataset_dir, action, sequence, hand_landmarks_data)
                csv_save(dataset_dir, action, sequence, hand_landmarks_data)


if __name__ == "__main__":
    dataset_generater()
