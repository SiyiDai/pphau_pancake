# https://github.com/IntelRealSense/librealsense/blob/81d469db173dd682d3bada9bd7c7570db0f7cf76/wrappers/python/examples/opencv_pointcloud_viewer.py
# License: Apache 2.0. See LICENSE file in root directory.
# Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

"""
OpenCV and Numpy Point cloud Software Renderer
This sample is mostly for demonstration and educational purposes.
It really doesn't offer the quality or performance that can be
achieved with hardware acceleration.
Usage:
------
Mouse:
    Drag with left button to rotate around pivot (thick small axes), 
    with right button to translate and the wheel to zoom.
Keyboard:
    [p]     Pause
    [r]     Reset View
    [d]     Cycle through decimation values
    [z]     Toggle point scaling
    [c]     Toggle color source
    [s]     Save PNG (./out.png)
    [e]     Export points to ply (./out.ply)
    [q\ESC] Quit
"""
import sys

sys.path.append("./Verb_detection/dataset_generation")
sys.path.append("./Verb_detection/model")
sys.path.append("./Verb_detection")
sys.path.append("./Noun_detection/Noun_detector")
sys.path.append("./Noun_detection/Noun_detector/yolo")
sys.path.append("./Noun_detection/Noun_detector/lib")
sys.path.append("./PPHAU")
print(sys.path)

import math
from re import T
import cv2
import numpy as np

# import torch
import copy
import rospy

# from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from sensor_msgs.msg import CameraInfo, CompressedImage
from sensor_msgs.msg import Image as Image_new
import message_filters
import cv_bridge
from utils.utils import *
import time
from verb_pipeline import detect_landmarks, process_verb
from scipy.ndimage import binary_dilation
from Noun_detection.Noun_detector.new_hand_obj_one_solution import *
from model import params
import numpy as np
import pickle


def check_is_valid_phrases(verb, nouns):
    valid_mapping = {
        "Close": ["Batter", "Batter cap", "Honey", "Rest"],
        "Decorate": [
            "Blueberry",
            "Pan with Pancake (0% finished)",
            "Pan with Pancake (50% finished)",
            "Pan with Pancake (80% finished)",
            "Pancake",
            "Plate",
            "Rest",
            "Spatula with Pancake (100% finished)",
            "Spatula with Pancake (50% finished)",
            "Spatula with Pancake (80% finished)",
        ],
        "Flip": [
            "Pan with Pancake (0% finished)",
            "Pan with Pancake (50% finished)",
            "Pan with Pancake (80% finished)",
            "Pan_handle",
            "Pancake",
            "Plate",
            "Rest",
            "Spatula with Pancake (100% finished)",
            "Spatula with Pancake (50% finished)",
            "Spatula with Pancake (80% finished)",
        ],
        "Move": [
            "Batter",
            "Batter cap",
            "Blueberry",
            "Honey",
            "Pan and Spatula",
            "Pan with Pancake (0% finished)",
            "Pan with Pancake (50% finished)",
            "Pan with Pancake (80% finished)",
            "Pan_handle",
            "Pancake",
            "Plate",
            "Raspberry",
            "Rest",
            "Spatula",
            "Spatula with Pancake (100% finished)",
            "Spatula with Pancake (50% finished)",
            "Spatula with Pancake (80% finished)",
        ],
        "Open": ["Batter", "Batter cap", "Honey"],
        "Pick up": [
            "Batter",
            "Batter cap",
            "Blueberry",
            "Honey",
            "Pan and Spatula",
            "Pan with Pancake (0% finished)",
            "Pan with Pancake (50% finished)",
            "Pan with Pancake (80% finished)",
            "Pan_handle",
            "Pancake",
            "Plate",
            "Raspberry",
            "Rest",
            "Spatula",
            "Spatula with Pancake (100% finished)",
            "Spatula with Pancake (50% finished)",
            "Spatula with Pancake (80% finished)",
        ],
        "Pour": [
            "Batter",
            "Batter cap",
            "Blueberry",
            "Honey",
            "Pan and Spatula",
            "Pan with Pancake (0% finished)",
            "Pan with Pancake (50% finished)",
            "Pan with Pancake (80% finished)",
            "Pan_handle",
            "Pancake",
            "Plate",
            "Raspberry",
            "Rest",
            "Spatula",
            "Spatula with Pancake (100% finished)",
            "Spatula with Pancake (50% finished)",
            "Spatula with Pancake (80% finished)",
        ],
        "Put down": [
            "Batter",
            "Batter cap",
            "Blueberry",
            "Honey",
            "Pan and Spatula",
            "Pan with Pancake (0% finished)",
            "Pan with Pancake (50% finished)",
            "Pan with Pancake (80% finished)",
            "Pan_handle",
            "Pancake",
            "Plate",
            "Raspberry",
            "Rest",
            "Spatula",
            "Spatula with Pancake (100% finished)",
            "Spatula with Pancake (50% finished)",
            "Spatula with Pancake (80% finished)",
        ],
        "Screw": ["Batter", "Batter cap", "Honey", "Rest"],
        "Shovel": [
            "Pan with Pancake (0% finished)",
            "Pan with Pancake (50% finished)",
            "Pan with Pancake (80% finished)",
            "Pan_handle",
            "Pancake",
            "Raspberry",
            "Rest",
            "Spatula with Pancake (100% finished)",
            "Spatula with Pancake (50% finished)",
            "Spatula with Pancake (80% finished)",
        ],
        "Squeeze": ["Batter", "Batter cap", "Honey"],
        "Other": [
            "Batter",
            "Batter cap",
            "Blueberry",
            "Honey",
            "Pan and Spatula",
            "Pan with Pancake (0% finished)",
            "Pan with Pancake (50% finished)",
            "Pan with Pancake (80% finished)",
            "Pan_handle",
            "Pancake",
            "Plate",
            "Raspberry",
            "Rest",
            "Spatula",
            "Spatula with Pancake (100% finished)",
            "Spatula with Pancake (50% finished)",
            "Spatula with Pancake (80% finished)",
        ],
        "Wait": [],
    }
    if len(nouns) >= 4:
        nouns = nouns[len(nouns) - 4 : len(nouns)]

    phrase = copy.deepcopy(verb)

    mapping = valid_mapping[verb]
    for noun in nouns:
        if noun not in mapping:
            continue

        phrase += " "
        phrase += noun
        break
    if verb == "Shovel" and len(list(phrase)) > 1:
        phrase = "Shovel Pancake"
    phrase += "..."
    return phrase


def preprocess_verb(verb):
    special_case = ["pick_up", "put_down"]
    if verb in special_case:
        verb = verb.split("_")
        verb = " ".join(verb)
    elif verb == "other":
        verb = "wait"
    elif verb == "move_object":
        verb = "move"
    verb = list(verb)
    verb[0] = verb[0].upper()

    return "".join(verb)


def overlay_function(image, mask, alpha=0.5):
    """
    image: input rgb image
    mask:  an image with same dimension as image, but class label in pixel-level
    """

    color_map = [
        [0, 0, 0],
        [255, 255, 50],
        [255, 50, 50],
        [50, 255, 50],
        [50, 50, 255],
        [255, 50, 255],
        [50, 255, 255],
    ]

    color_map_np = np.array(color_map)

    """ Overlay segmentation on top of RGB image """
    im_overlay = image.copy()

    colored_mask = color_map_np[mask]
    foreground = image * alpha + (1 - alpha) * colored_mask
    binary_mask = mask > 0
    # Compose image
    im_overlay[binary_mask] = foreground[binary_mask]
    countours = binary_dilation(binary_mask) ^ binary_mask
    im_overlay[countours, :] = 0
    return im_overlay.astype(image.dtype)


def get_full_mask(object_masks, mask, bbox, indices):
    k = 0
    output = mask
    for i in indices:
        x, y, w, h = bbox[i]
        output[y : y + h, x : x + w] = object_masks[k]
        k += 1
    return output


conf_threshold = 0.5
nms_threshold = 0.3


def get_objects(outputs, img):
    hT, wT, cT = img.shape
    objects = []
    bbox = []
    confs = []
    for detection in outputs:
        #         for detection in output:
        scores = detection[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]

        if confidence > conf_threshold:
            w, h = int(detection[2]), int(detection[3])
            # Center point of the object
            x, y = int(detection[0]), int(detection[1])
            bbox.append([x, y, w, h])
            object_temp = img[y : y + h, x : x + w, :]
            objects.append(object_temp)
            confs.append(float(confidence))
    indices = cv2.dnn.NMSBoxes(bbox, confs, conf_threshold, nms_threshold)
    return objects, indices, bbox


# Modify:

# process_color_depth self.depth


class ROSDataSource:
    def __init__(
        self,
        color_topic="/camera/color/image_raw/compressed",
        # color_camera_info_topic="/camera/142122070031/color/camera_info",
        depth_topic="/camera/142122070031/aligned_depth_to_color/image_raw",
        depth_camera_info_topic="/camera/142122070031/aligned_depth_to_color/camera_info",
    ):
        self.cv_bridge = cv_bridge.CvBridge()
        # self.model = model
        rospy.init_node("multiview", anonymous=True)
        self.color_subscriber = message_filters.Subscriber(color_topic, CompressedImage)
        self.depth_subscriber = message_filters.Subscriber(depth_topic, Image_new)
        self.depth_camera_info_subscriber = message_filters.Subscriber(
            depth_camera_info_topic, CameraInfo
        )

        self.synced_color_depth = message_filters.ApproximateTimeSynchronizer(
            [self.color_subscriber, self.depth_subscriber], queue_size=10, slop=0.5
        )

        self.color_camera_info = None
        self.K = None
        self.Kinv = None
        self.color_image = None
        self.depth_image = None
        self.height = 480
        self.width = 640

        self.pixel_coords = (
            np.mgrid[: self.height, : self.width].transpose(1, 2, 0).reshape(-1, 2)
        )
        self.float_pixel_coords = np.array(
            np.mgrid[: self.height, : self.width].transpose(1, 2, 0), dtype=np.float32
        )
        self.texcoords = self.float_pixel_coords.copy()
        # self.texcoords[:,:,0] = (self.height - self.texcoords[:,:,0])/self.height
        self.texcoords[:, :, 0] = self.texcoords[:, :, 0] / self.height
        self.texcoords[:, :, 1] = self.texcoords[:, :, 1] / self.width
        self.texcoords = self.texcoords[:, :, ::-1]
        self.texcoords = self.texcoords.reshape(-1, 2)
        self.num_points = self.pixel_coords.shape[0]
        self.hpixel_coords = np.concatenate(
            (self.pixel_coords, np.ones((self.num_points, 1))), axis=1
        )
        self.rows = self.pixel_coords[:, 0]
        self.cols = self.pixel_coords[:, 1]
        self.proj_mat = None
        print("initialized..")

    def process_color_camera_info(self, color_camera_info):
        self.color_camera_info = color_camera_info

    def process_depth_camera_info(self, depth_camera_info):
        # print("received depth camera info")
        if self.K is None:
            self.K = np.array(depth_camera_info.K).reshape((3, 3))
            self.Kinv = np.linalg.inv(self.K)
            self.proj_mat = self.hpixel_coords @ self.Kinv.T

    def process_color_depth(self, ci, di=0):
        # print(ci is not None, di is not None)
        # import pdb; pdb.set_trace()
        self.color_image = self.cv_bridge.compressed_imgmsg_to_cv2(ci)
        self.color_image = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB)

        self.depth_image = self.cv_bridge.imgmsg_to_cv2(di, "16UC1")

    def start(self):
        self.synced_color_depth.registerCallback(self.process_color_depth)
        self.depth_camera_info_subscriber.registerCallback(
            self.process_depth_camera_info
        )
        print("started..")

    def write_image(self):
        folder = "./data"
        self.start()
        i = 0
        while True:
            color_image, depth_image, K, pcd = self.next()
            if color_image is not None:
                path = folder + "/data" + str(i) + ".pkl"
                dict = {
                    "color_image": color_image,
                    "depth_image": depth_image,
                    "K": K,
                    "pcd": pcd,
                }
                with open(path, "wb") as f:
                    pickle.dump(dict, f)

                i += 1
            print(type(color_image))
            print(type(depth_image))
            print(type(K))
            print(type(pcd))

    def next(self):
        if self.color_image is None or self.depth_image is None or self.K is None:
            return None, None, None, None

        color_image = self.color_image  # .copy()
        depth_image = self.depth_image  # .copy()

        K = self.K
        now = time.time()
        pcd = self.point_cloud(depth_image, self.Kinv)
        # print("Process point cloud:", time.time() - now)
        return color_image, depth_image, K, pcd

    def point_cloud(self, depth_image, Kinv):
        if depth_image is None or Kinv is None:
            # import pdb; pdb.set_trace()
            return None

        Z = depth_image.reshape((-1, 1)) / 1000
        # import pdb; pdb.set_trace()
        return (self.hpixel_coords @ Kinv.T) * Z

    @property
    def pcd(self):
        return self.point_cloud(self.depth_image, self.Kinv)

    def close(self):
        return


class AppState:
    def __init__(self, *args, **kwargs):
        self.WIN_NAME = "RealSense"
        self.pitch, self.yaw = math.radians(-10), math.radians(-15)
        self.translation = np.array([0, 0, -1], dtype=np.float32)
        self.distance = 2
        self.prev_mouse = 0, 0
        self.mouse_btns = [False, False, False]
        self.paused = False
        self.decimate = 1
        self.scale = True
        self.color = True

    def reset(self):
        self.pitch, self.yaw, self.distance = 0, 0, 2
        self.translation[:] = 0, 0, -1

    @property
    def rotation(self):
        Rx, _ = cv2.Rodrigues((self.pitch, 0, 0))
        Ry, _ = cv2.Rodrigues((0, self.yaw, 0))
        return np.dot(Ry, Rx).astype(np.float32)

    @property
    def pivot(self):
        return self.translation + np.array((0, 0, self.distance), dtype=np.float32)


class Visualizer:
    def __init__(self, data_source):
        self.state = AppState()
        self.data_source = data_source
        self.w = self.data_source.width
        self.h = self.data_source.height

    def mouse_cb(self, event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:
            self.state.mouse_btns[0] = True

        if event == cv2.EVENT_LBUTTONUP:
            self.state.mouse_btns[0] = False

        if event == cv2.EVENT_RBUTTONDOWN:
            self.state.mouse_btns[1] = True

        if event == cv2.EVENT_RBUTTONUP:
            self.state.mouse_btns[1] = False

        if event == cv2.EVENT_MBUTTONDOWN:
            self.state.mouse_btns[2] = True

        if event == cv2.EVENT_MBUTTONUP:
            self.state.mouse_btns[2] = False

        if event == cv2.EVENT_MOUSEMOVE:

            h, w = self.out.shape[:2]
            dx, dy = x - self.state.prev_mouse[0], y - self.state.prev_mouse[1]

            if self.state.mouse_btns[0]:
                self.state.yaw += float(dx) / w * 2
                self.state.pitch -= float(dy) / h * 2

            elif self.state.mouse_btns[1]:
                dp = np.array((dx / w, dy / h, 0), dtype=np.float32)
                self.state.translation -= np.dot(self.state.rotation, dp)

            elif self.state.mouse_btns[2]:
                dz = math.sqrt(dx ** 2 + dy ** 2) * math.copysign(0.01, -dy)
                self.state.translation[2] += dz
                self.state.distance -= dz

        if event == cv2.EVENT_MOUSEWHEEL:
            dz = math.copysign(0.1, flags)
            self.state.translation[2] += dz
            self.state.distance -= dz

        self.state.prev_mouse = (x, y)

    def start(self):
        cv2.namedWindow(self.state.WIN_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.resizeWindow(self.state.WIN_NAME, self.w, self.h)
        cv2.setMouseCallback(self.state.WIN_NAME, self.mouse_cb)
        self.data_source.start()

    def project(self, v):
        """project 3d vector array to 2d"""

        h, w = self.out.shape[:2]
        assert h == self.h and w == self.w
        view_aspect = float(h) / w

        # ignore divide by zero for invalid depth
        with np.errstate(divide="ignore", invalid="ignore"):
            proj = v[:, :-1] / v[:, -1, np.newaxis] * (w * view_aspect, h) + (
                w / 2.0,
                h / 2.0,
            )

        # near clipping
        znear = 0.03
        proj[v[:, 2] < znear] = np.nan
        return proj

    def review(self, pcd, color, R, t):

        pcd = pcd @ R + t

    def view(self, v):
        """apply view transformation on vector array"""
        return (
            np.dot(v - self.state.pivot, self.state.rotation)
            + self.state.pivot
            - self.state.translation
        )

    def line3d(self, out, pt1, pt2, color=(0x80, 0x80, 0x80), thickness=1):
        """draw a 3d line from pt1 to pt2"""
        p0 = self.project(pt1.reshape(-1, 3))[0]
        p1 = self.project(pt2.reshape(-1, 3))[0]
        if np.isnan(p0).any() or np.isnan(p1).any():
            return
        p0 = tuple(p0.astype(int))
        p1 = tuple(p1.astype(int))
        rect = (0, 0, out.shape[1], out.shape[0])
        inside, p0, p1 = cv2.clipLine(rect, p0, p1)
        if inside:
            cv2.line(out, p0, p1, color, thickness, cv2.LINE_AA)

    def grid(
        self, out, pos, rotation=np.eye(3), size=1, n=10, color=(0x80, 0x80, 0x80)
    ):
        """draw a grid on xz plane"""
        pos = np.array(pos)
        s = size / float(n)
        s2 = 0.5 * size
        for i in range(0, n + 1):
            x = -s2 + i * s
            self.line3d(
                out,
                self.view(pos + np.dot((x, 0, -s2), rotation)),
                self.view(pos + np.dot((x, 0, s2), rotation)),
                color,
            )
        for i in range(0, n + 1):
            z = -s2 + i * s
            self.line3d(
                out,
                self.view(pos + np.dot((-s2, 0, z), rotation)),
                self.view(pos + np.dot((s2, 0, z), rotation)),
                color,
            )

    def axes(self, out, pos, rotation=np.eye(3), size=0.075, thickness=2):
        """draw 3d axes"""
        self.line3d(
            out, pos, pos + np.dot((0, 0, size), rotation), (0xFF, 0, 0), thickness
        )
        self.line3d(
            out, pos, pos + np.dot((0, size, 0), rotation), (0, 0xFF, 0), thickness
        )
        self.line3d(
            out, pos, pos + np.dot((size, 0, 0), rotation), (0, 0, 0xFF), thickness
        )

    def frustum(self, out, intrinsics, pcd, color=(0x40, 0x40, 0x40)):
        """draw camera's frustum"""
        orig = self.view([0, 0, 0])
        # w, h = intrinsics.width, intrinsics.height

        for d in range(1, 6, 2):

            def get_point(x, y):
                # import pdb; pdb.set_trace()
                p = pcd[x * self.w + y]
                # p = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], d)
                self.line3d(out, orig, self.view(p), color)
                return p

            top_left = get_point(0, 0)
            top_right = get_point(self.w, 0)
            bottom_right = get_point(self.w, self.h)
            bottom_left = get_point(0, self.h)

            self.line3d(out, self.view(top_left), self.view(top_right), color)
            self.line3d(out, self.view(top_right), self.view(bottom_right), color)
            self.line3d(out, self.view(bottom_right), self.view(bottom_left), color)
            self.line3d(out, self.view(bottom_left), self.view(top_left), color)

    def pointcloud(self, out, verts, texcoords, color, painter=True):
        # painter = False
        """draw point cloud with optional painter's algorithm"""
        if painter:
            # Painter's algo, sort points from back to front

            # get reverse sorted indices by z (in view-space)
            # https://gist.github.com/stevenvo/e3dad127598842459b68
            v = self.view(verts)
            s = v[:, 2].argsort()[::-1]
            proj = self.project(v[s])
        else:
            proj = self.project(self.view(verts))

        if self.state.scale:
            proj *= 1.0 ** self.state.decimate

        h, w = out.shape[:2]

        # proj now contains 2d image coordinates
        j, i = proj.astype(np.uint32).T

        # create a mask to ignore out-of-bound indices
        im = (i >= 0) & (i < h)
        jm = (j >= 0) & (j < w)
        m = im & jm

        cw, ch = color.shape[:2][::-1]
        if painter:
            # sort texcoord with same indices as above
            # texcoords are [0..1] and relative to top-left pixel corner,
            # multiply by size and add 0.5 to center
            v, u = (texcoords[s] * (cw, ch) + 0.5).astype(np.uint32).T
        else:
            v, u = (texcoords * (cw, ch) + 0.5).astype(np.uint32).T
        # clip texcoords to image
        np.clip(u, 0, ch - 1, out=u)
        np.clip(v, 0, cw - 1, out=v)
        # import pdb; pdb.set_trace()
        # perform uv-mapping
        out[i[m], j[m]] = color[u[m], v[m]]

    def loop(self):
        self.out = np.empty((self.h, self.w, 3), dtype=np.uint8)
        i = 0
        color_image_batch = []
        verb = "Wait"
        # Load yolo weights
        className, net = yolo_precfg(
            path_classesFile="./Object_detection/obj.names",
            path_modelWeights="./Object_detection/best.weights",
            path_modelcfg="./Object_detection/yolov3_test.cfg",
        )
        print("Loading Yolo weights successfully")
        # Load faster RCNN model
        fasterRCNN, pascal_classes = load_model()
        # Load efficientNet
        num_classes = 17
        eff_model = create_model(num_classes=num_classes)
        weights_path = "./Noun_detection/Noun_detector/one_solution_efficient/save_weights/efficientnet.h5"
        assert len(glob.glob(weights_path + "*")), "cannot find {}".format(weights_path)

        eff_model.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print("Loading efficientNet successfully")
        # Load LSTM model
        LSTM_model_name = "./Verb_detection/finalized_model.h5"
        # LSTM_model = pickle.load(open(LSTM_model_name, "rb"))
        LSTM_model = tf.keras.models.load_model(LSTM_model_name)
        # print(LSTM_model.summary())
        print("Loading LSTM successfully")
        print("All models are loaded!")
        # Load verb classes
        labels = params.classes
        verb_array = []
        verb_file_name = "verb_array.pkl"
        nouns_array = ["Batter"]
        nouns_file_name = "nouns_array.pkl"
        # Start loop
        while True:
            # Grab camera data
            if not self.state.paused:
                (
                    color_image,
                    depth_image,
                    depth_intrinsics,
                    points,
                ) = self.data_source.next()

                if color_image is not None:
                    # Make a copy
                    temp_color_image = copy.deepcopy(color_image)
                    temp_color_image = cv2.cvtColor(temp_color_image, cv2.COLOR_RGB2BGR)
                    # Yolo processing
                    print("*" * 50)
                    color_image_yolo = copy.deepcopy(color_image)

                    now = time.time()
                    process_yolo(color_image_yolo, className, net)
                    print("Process yolo(s):", time.time() - now)
                    color_image_yolo = cv2.cvtColor(color_image_yolo, cv2.COLOR_RGB2BGR)

                    # Mediapipe processing
                    print("*" * 50)
                    now = time.time()
                    temp, annotated_image1 = detect_landmarks(color_image, 1)
                    print("Process mediapipe(s):", time.time() - now)
                    annotated_image1 = cv2.cvtColor(annotated_image1, cv2.COLOR_RGB2BGR)
                    # Noun processing
                    color_image_noun = copy.deepcopy(color_image)
                    print("*" * 50)
                    now = time.time()
                    color_image_noun, nouns = hand_obj_one_solution(
                        color_image_noun, eff_model, fasterRCNN, pascal_classes
                    )

                    # if len(nouns) > 0:
                    #     for noun in nouns:
                    #         nouns_array.append(noun)
                    #     noun_count = 0
                    #     for noun in reversed(nouns_array):
                    #         nouns.append(noun)
                    #         noun_count+=1
                    #         if noun_count>=10:
                    #             break
                    # else:
                    #     nouns = copy.deepcopy(nouns_array)
                    print(nouns)
                    color_image_noun = cv2.cvtColor(color_image_noun, cv2.COLOR_RGB2BGR)

                    print("Process noun(s):", time.time() - now)
                    print("Nouns:", nouns)
                    print("*" * 50)

                    # Verb processing
                    color_image_verb = copy.deepcopy(color_image)
                    if i < 9:
                        color_image_batch.append(
                            [i, color_image_verb]
                        )  # RGB color image

                    elif i >= 9:
                        color_image_batch.append([9, color_image_verb])
                        now = time.time()
                        _, verb = process_verb(color_image_batch, LSTM_model, labels)
                        verb_array.append(verb)
                        open_file = open(verb_file_name, "wb")
                        pickle.dump(verb_array, open_file)
                        open_file.close()

                        print("*" * 50)
                        print("Verb processing(s):", time.time() - now)
                        print("Verb:", verb)
                        # now = time.time()
                        del color_image_batch[0]
                        for j in range(len(color_image_batch)):
                            color_image_batch[j][0] -= 1
                            # print(color_image_batch[j][0])
                        # print("Image batch append(s):", time.time() - now)
                    i += 1

                    # Phrase processing
                    # Preprocess verb
                    verb = preprocess_verb(verb)
                    # Preprocess phrase

                    phrase = check_is_valid_phrases(verb, nouns)

                    cv2.putText(
                        img=color_image_verb,
                        text=phrase,
                        org=(10, 100),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1.2,
                        color=(255, 64, 64),
                        thickness=2,
                    )
                    color_image_verb = cv2.cvtColor(color_image_verb, cv2.COLOR_RGB2BGR)

                # Pointcloud data to arrays
                # if points is None:
                #     continue
                if color_image is None:
                    continue
                color_image = copy.deepcopy(color_image_noun)

                texcoords = self.data_source.texcoords
                verts = points.reshape((-1, 3))

            # Render

            self.out.fill(0)

            # self.grid(self.out, (0, 0.5, 1), size=1, n=10)
            # self.frustum(self.out, depth_intrinsics, points)
            self.axes(
                self.out,
                self.view([0, 0, 0]),
                self.state.rotation,
                size=0.1,
                thickness=1,
            )

            if not self.state.scale or self.out.shape[:2] == (self.h, self.w):
                self.pointcloud(self.out, verts, texcoords, color_image)
                # import pdb; pdb.set_trace()
            else:
                tmp = np.zeros((self.h, self.w, 3), dtype=np.uint8)
                self.pointcloud(tmp, verts, texcoords, color_image)
                tmp = cv2.resize(
                    tmp, self.out.shape[:2][::-1], interpolation=cv2.INTER_NEAREST
                )
                np.putmask(self.out, tmp > 0, tmp)

            if any(self.state.mouse_btns):
                self.axes(
                    self.out,
                    self.view(self.state.pivot),
                    self.state.rotation,
                    thickness=4,
                )

            cv2.setWindowTitle(
                self.state.WIN_NAME, "PPHAU"
            )  #  % (self.w, self.h, 1.0/dt, dt*1000, "PAUSED" if self.state.paused else ""))

            # cv2.imshow("color", color_image)
            first_row = np.concatenate([self.out, color_image_yolo], axis=1)
            pancake_img = cv2.imread("pancake.jpg")
            dim = (self.data_source.width, self.data_source.height)
            resized_pancake = cv2.resize(pancake_img, dim)
            resized_pancake[:, :, :] = 0
            # resized_pancake = np.zeros([dim[1], dim[0], 3])
            cv2.putText(
                img=resized_pancake,
                text=phrase,
                org=(10, 100),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.2,
                color=(255, 255, 255),
                thickness=2,
            )
            cv2.putText(
                img=resized_pancake,
                text="Group B: Perfect Pancake",
                org=(100, 200),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.2,
                color=(255, 255, 255),
                thickness=2,
            )
            cv2.putText(
                img=resized_pancake,
                text="Siyi Dai, Shuang Wang",
                org=(110, 280),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.2,
                color=(255, 255, 255),
                thickness=2,
            )
            cv2.putText(
                img=resized_pancake,
                text="Zheng Tao, Chi Wang",
                org=(120, 320),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.2,
                color=(255, 255, 255),
                thickness=2,
            )
            second_row = np.concatenate(
                [
                    annotated_image1
                    + color_image_verb
                    + color_image_noun
                    - 2 * temp_color_image,
                    resized_pancake,
                ],
                axis=1,
            )
            image_for_display = np.concatenate([first_row, second_row], axis=0)
            cv2.imshow(self.state.WIN_NAME, image_for_display)

            key = cv2.waitKey(1)

            if key == ord("q"):
                cv2.destroyAllWindows()
                break

        self.close()

    def close(self):
        # Stop streaming
        self.data_source.close()


if __name__ == "__main__":
    # ds = ROSDataSource(color_topic="camera/color/image_raw",
    #  color_camera_info_topic="/camera/color/camera_info",
    #  depth_topic="/camera/aligned_depth_to_color/image_raw",
    #  depth_camera_info_topic="/camera/aligned_depth_to_color/camera_info",)

    import tensorflow as tf

    print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))
    ds = ROSDataSource(
        color_topic="/camera/color/image_raw/compressed",
        depth_topic="/camera/aligned_depth_to_color/image_raw",
        depth_camera_info_topic="/camera/aligned_depth_to_color/camera_info",
    )

    vis = Visualizer(ds)
    print("work")
    vis.start()
    vis.loop()
