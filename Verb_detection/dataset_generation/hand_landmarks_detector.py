import os
import cv2
import mediapipe as mp
import pandas as pd

from actions_dataset_io import save_hand_landmarks_image, json_struct


VISUALIZE = False

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles


def mp_process(image):
    with mp_hands.Hands(
        static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5
    ) as hands:
        results = hands.process(image)
    return results


def get_result_multi_handedness(results, debug=False):
    if debug:
        # Print handedness and draw hand landmarks on the image.
        print("Handedness:", results.multi_handedness)

    # manage multi handedness
    result_multi_handedness = []
    for hand in results.multi_handedness:
        result_multi_handedness.append(str(hand))
    return result_multi_handedness


def get_result_multi_landmarks_json(results):
    # manage multi hand landmarks
    result_multi_landmarks = []
    for hand_landmarks in results.multi_hand_landmarks:
        keypoints = []
        for data_point in hand_landmarks.landmark:
            keypoints.append(
                {
                    "X": data_point.x,
                    "Y": data_point.y,
                    "Z": data_point.z,
                }
            )
        result_multi_landmarks.append(keypoints)
    return result_multi_landmarks


def get_result_multi_landmarks(results, file, action):
    ts = get_ts(file)
    data = {"timestamp": ts, "action": action}
    for idx_hl, hand_landmarks in enumerate(results.multi_hand_landmarks):
        classification = list(results.multi_handedness[idx_hl].classification)
        # 0: Left, 1: Right
        hand = get_hand(classification)
        for idx_dp, data_point in enumerate(hand_landmarks.landmark):
            data[f"{hand}_{idx_dp}_x"] = data_point.x
            data[f"{hand}_{idx_dp}_y"] = data_point.y
            data[f"{hand}_{idx_dp}_z"] = data_point.z
    # check how many hands were detected, if only one hand was detected, fill the other as 0
    if len(data) == 21 * 3 + 2:
        idx_hl_fill = 1 - int(hand)
        for idx_dp in range(21):
            data[f"{idx_hl_fill}_{idx_dp}_x"] = 0.0
            data[f"{idx_hl_fill}_{idx_dp}_y"] = 0.0
            data[f"{idx_hl_fill}_{idx_dp}_z"] = 0.0

    data_pd = pd.DataFrame(data, index=[ts])
    return data_pd


def get_ts(file):
    num = "".join(filter(lambda i: i.isdigit(), file))
    return int(num)


def get_hand(classification):
    emp_lis = []
    for z in str(classification).split():
        if z.isdigit():
            emp_lis.append(int(z))
    return emp_lis[0]


def visualize_hand_landmarks(
    sequence_folder,
    file,
    image,
    results,
    debug=False,
    save_hand_landmarks=False,
):
    image_height, image_width, _ = image.shape
    # visualize the hand landmarks to image
    annotated_image = image.copy()
    for hand_landmarks in results.multi_hand_landmarks:
        if debug:
            print("hand_landmarks:", hand_landmarks)
            print(
                f"Index finger tip coordinates: (",
                f"{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, "
                f"{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})",
            )
        mp_drawing.draw_landmarks(
            annotated_image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            # mp_drawing_styles.get_default_hand_landmarks_style(),
            # mp_drawing_styles.get_default_hand_connections_style(),
        )
        if save_hand_landmarks:
            save_hand_landmarks_image(sequence_folder, file, annotated_image)

        # Debug
        while debug == True:
            cv2.imshow("mediapipe_hand_image", annotated_image)
            key = cv2.waitKey(1)
            # if pressed escape exit program
            if key == 27:
                cv2.destroyAllWindows()
                break


def detect_landmarks(image_folder, file, action, sequence_folder):
    image = cv2.imread(os.path.join(image_folder, file))
    results = mp_process(image)
    if results.multi_hand_landmarks:
        result_multi_landmarks = get_result_multi_landmarks(
            results, file, action
        )

        if VISUALIZE:
            visualize_hand_landmarks(sequence_folder, file, image, results)

        return result_multi_landmarks


def detect_results(image_folder, file, action):
    image = cv2.imread(os.path.join(image_folder, file))
    results = mp_process(image)
    if results.multi_hand_landmarks:
        result_multi_handedness = get_result_multi_handedness(results)
        result_multi_landmarks = get_result_multi_landmarks(
            results, file, action
        )

        if VISUALIZE:
            visualize_hand_landmarks(file, image, results)

        output = json_struct(
            file, result_multi_handedness, result_multi_landmarks
        )

        return output


# def main():
#     dataset_dir = DATASET_DIR
#     action_list = read_action(dataset_dir)

#     for _, action in enumerate(action_list):
#         sequence_index_list = read_sequence_index(dataset_dir, action)

#         for _, sequence in enumerate(sequence_index_list):
#             image_folder, image_files, sequence_folder = read_image_files(
#                 dataset_dir, action, sequence
#             )

#             # if json_exist(join(dataset_dir, action, sequence)):
#             if csv_exist(join(dataset_dir, action, sequence)):
#                 # if json already saved, continue
#                 continue

#             hand_landmarks_data = []
#             for _, file in enumerate(image_files):
#                 # output = detect_results(image_folder, file,action)
#                 output = detect_landmarks(image_folder, file, action, sequence_folder)
#                 hand_landmarks_data.append(output)
#             if hand_landmarks_data:
#                 # json_save(dataset_dir, action, sequence, hand_landmarks_data)
#                 csv_save(dataset_dir, action, sequence, hand_landmarks_data)


# if __name__ == "__main__":
#     main()
