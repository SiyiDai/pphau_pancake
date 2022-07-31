import time
import numpy as np
# from Active_object_upload.hand_obj_one_solution import *
import os
import cv2
def yolo_precfg(**kwargs):
    # set the initial path
    # set classes file
    if 'path_classesFile' in kwargs.keys():
        classesFile = kwargs['path_classesFile']
    else:
        classesFile = os.path.join(os.getcwd(), 'data/obj.names')
    # set model configuration
    if 'path_modelcfg' in kwargs.keys():
        modelConfiguration = kwargs['path_modelcfg']
    else:
        classesFile = os.path.join(os.getcwd(), 'data/yolov3_test.cfg')
    # set model weights
    if 'path_modelWeights' in kwargs.keys():
        modelWeights = kwargs['path_modelWeights']
    else:
        classesFile = os.path.join(os.getcwd(), 'weights/yolov3_best.weights')

    with open(classesFile, 'rt') as f:
        className = f.read().rstrip('\n').split('\n')

    # set YOLO-v3 network
    net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    return className, net


def find_objects(outputs, img, className):
    conf_threshold = 0.3
    nms_threshold = 0.3

    hT, wT, cT = img.shape
    bbox, confs, classIDs = [], [], []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > conf_threshold:
                w, h = int(detection[2] * wT), int(detection[3] * wT)
                x, y = int(detection[0] * wT - w / 2), int(detection[1] * hT - h / 2)
                bbox.append([x, y, w, h])
                classIDs.append(classID)
                confs.append(float(confidence))
                # print(classIDs)

    # print(len(bbox))
    indices = cv2.dnn.NMSBoxes(bbox, confs, conf_threshold, nms_threshold)

    for object_i in indices:
        # print(object_i, 'object')
        # i = object_i[0]
        i = object_i
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, f'{className[classIDs[i]].upper()}{int(confs[i] * 100)}%',
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

def process_yolo(color_image, className, net):
    # whT = (640, 480)
    whT = 256
    blob = cv2.dnn.blobFromImage(color_image, 1/255.0, (whT,whT), crop = False)
    net.setInput(blob)

    layernames = net.getLayerNames()
    first = time.time()
    outputnames = [(layernames[i-1]) for i in net.getUnconnectedOutLayers()]
    second = time.time()
    outputs = net.forward(outputnames)
    third = time.time()
    find_objects(outputs, color_image, className)
    fourth = time.time()
# def process_active_object(color, fasterRCNN, pascal_classes):
#     return hand_obj_one_solution(color, fasterRCNN, pascal_classes)



