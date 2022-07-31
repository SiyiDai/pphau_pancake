#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 22:49:14 2022


"""

import cv2
import os
import numpy as np
from matplotlib import pyplot as plt


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
        
    with open(classesFile,'rt') as f:
        className = f.read().rstrip('\n').split('\n')
    
    # set YOLO-v3 network
    net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    
    return className, net

# function for find objects and draw bounding boxes
def find_objects(outputs, img):
    # Threshold settings
    conf_threshold = 0.5
    nms_threshold = 0.3
    
    hT, wT, cT = img.shape
    bbox, confs, classIDs = [], [], []
    inference_per_frame = {}

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > conf_threshold:
                w, h = int(detection[2]*wT), int(detection[3]*wT)
                x, y = int(detection[0]*wT - w/2), int(detection[1]*hT - h/2)
                bbox.append([x, y, w, h])
                classIDs.append(classID)
                confs.append(float(confidence))
                # print(classIDs)
                
                bbox_info = np.mat([[x,y],
                                    [w,h]])
                obj_name =  className[classID]
                class_info = {obj_name: {"bbox": bbox_info, "conf": confidence}}
                inference_per_frame.update(class_info)
                

    # print(len(bbox))
    indices = cv2.dnn.NMSBoxes(bbox, confs, conf_threshold, nms_threshold)
    
    for object_i in indices:
        # print(object_i, 'object')
        # i = object_i[0]
        i = object_i
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, f'{className[classIDs[i]].upper()}{int(confs[i]*100)}%',
                   (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
    return inference_per_frame
        
if __name__ == "__main__":
    
    img_path = "/home/kolz14w/Proj_Master/haucode/proj/anno_yolo/pphau_group_b/darknet/data/obj_train_data/left0300cw.jpg"
    vid_path = "/home/kolz14w/Proj_Master/haucode/proj/anno_yolo/pphau_group_b/darknet/data/test/output.mp4"
    input_type = "video"
    
    
    # for speeding up the detection, you could modify whT to a smaller value,
    # but it must be divisible by 32.
    # Note that if whT was too small, the output could muss WORSE than the expectation.
    whT = 416
    
    # Initialize the DNN and corresponding classname
    
    className, net = yolo_precfg(path_classesFile = "/home/kolz14w/Proj_Master/haucode/proj/anno_yolo/pphau_group_b/darknet/data_fine_tuning/obj.names",
                                 path_modelWeights = "/home/kolz14w/Proj_Master/haucode/proj/anno_yolo/pphau_group_b/darknet/weights/yolov3_last.weights", 
                                 path_modelcfg = "/home/kolz14w/Proj_Master/haucode/proj/anno_yolo/pphau_group_b/darknet/data/yolov3_test_14.cfg")
    if input_type == "image":
        # Single image obeject detection
        img = cv2.imread(img_path)        
        
        blob = cv2.dnn.blobFromImage(img, 1/255.0, (whT, whT), swapRB = True, crop = False)
        net.setInput(blob)
        
        layernames = net.getLayerNames()
        outputnames = [(layernames[i-1]) for i in net.getUnconnectedOutLayers()]
        # print(outputnames)
        outputs = net.forward(outputnames)
        # print(outputs)
        inference_per_frame = find_objects(outputs, img)
        plt.imshow(img)
    
    
    elif input_type == "video":
        # Video object detection
        vid = cv2.VideoCapture(vid_path)
        # vid = cv2.VideoCapture(0)
        c = 0
        if vid.isOpened():
            rval, frame = vid.read()
        else:
            print('Openerror! ')
            rval = False
        num_frame = int(vid.get(7))       # get the total number of frames in the selected video
        for i in range(num_frame-1):
            rval, img = vid.read()
            c += 1
            blob = cv2.dnn.blobFromImage(img, 1/255.0, (whT, whT), swapRB = True, crop = False)
            net.setInput(blob)
            
            layernames = net.getLayerNames()
            outputnames = [(layernames[i-1]) for i in net.getUnconnectedOutLayers()]
            # print(outputnames)
            outputs = net.forward(outputnames)
            # print(outputs)
            inference_per_frame = find_objects(outputs, img)
            cv2.imshow("video_stream", img)
            key = cv2.waitKey(1)
            if key == 27: # pressed escape
                cv2.destroyAllWindows()
                break
        