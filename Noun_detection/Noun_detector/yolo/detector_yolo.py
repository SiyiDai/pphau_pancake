#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 22:49:14 2022


"""

import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

import pandas as pd

import pickle


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
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    
    return className, net

# function for find objects and draw bounding boxes
def find_objects(className, outputs, img):
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

def YOLO():
    img_path = "../image"
    vid_path = "./data/test/output.mp4"
    
    whT = 416
    
    # Single image obeject detection
    # img = cv2.imread(img_path)        
    
    className, net = yolo_precfg(path_classesFile = "./data/obj.names",
                                 path_modelWeights = "./weights/yolov3_best.weights", 
                                 path_modelcfg = "./data/yolov3_test.cfg")
    
    # blob = cv2.dnn.blobFromImage(img, 1/255.0, (whT, whT), swapRB = True, crop = False)
    # net.setInput(blob)
    
    # layernames = net.getLayerNames()
    # outputnames = [(layernames[i-1]) for i in net.getUnconnectedOutLayers()]
    # # print(outputnames)
    # outputs = net.forward(outputnames)
    # # print(outputs)
    # inference_per_frame = find_objects(outputs, img)
    # plt.imshow(img)
    
    
    # Video object detection
    vid = cv2.VideoCapture(vid_path)
    # vid = cv2.VideoCapture(0)
    c = 0
    if vid.isOpened():
        print('video mode! ')
        rval, frame = vid.read()
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
            inference_per_frame = find_objects(className, outputs, img)
            cv2.imshow("video_stream", img)
            key = cv2.waitKey(1)
            if key == 27: # pressed escape
                cv2.destroyAllWindows()
                break
    else:
        print('image mode! ')
        rval = False
        inference_per_frame = []
        for image in os.listdir(img_path):
            
            path = os.path.join(img_path, image)
        
            if image.endswith('png'):
                img = cv2.imread(path)[...,::-1]
            else:
                img = cv2.imread(path)
                
            blob = cv2.dnn.blobFromImage(img, 1/255.0, (whT, whT), swapRB = True, crop = False)
            net.setInput(blob)
            
            layernames = net.getLayerNames()
            outputnames = [(layernames[i-1]) for i in net.getUnconnectedOutLayers()]
            # print(outputnames)
            outputs = net.forward(outputnames)
            # print(outputs)
            inference_per_frame.append(find_objects(className, outputs, img))
            cv2.imshow("a single image", img)
            key = cv2.waitKey(0) # '0' means waiting forever
            if key == ord('q'): # pressed escape
                cv2.destroyAllWindows()
    
    # df = pd.DataFrame(inference_per_frame)
    # df.to_csv('./inference_per_frame.csv', index=False)
    
    with open('./inference_per_frame.pkl', 'wb') as f:
        pickle.dump(inference_per_frame, f)
        

def YOLO_not_write_in_file(img_path):
    # img_path = "../image"
    vid_path = "./data/test/output.mp4"
    
    whT = 416
    
    # Single image obeject detection
    # img = cv2.imread(img_path)        
    
    className, net = yolo_precfg(path_classesFile = "./data/obj.names",
                                 path_modelWeights = "./weights/yolov3_best.weights", 
                                 path_modelcfg = "./data/yolov3_test.cfg")
    
    # blob = cv2.dnn.blobFromImage(img, 1/255.0, (whT, whT), swapRB = True, crop = False)
    # net.setInput(blob)
    
    # layernames = net.getLayerNames()
    # outputnames = [(layernames[i-1]) for i in net.getUnconnectedOutLayers()]
    # # print(outputnames)
    # outputs = net.forward(outputnames)
    # # print(outputs)
    # inference_per_frame = find_objects(outputs, img)
    # plt.imshow(img)
    
    
    # Video object detection
    vid = cv2.VideoCapture(vid_path)
    # vid = cv2.VideoCapture(0)
    c = 0
    if vid.isOpened():
        print('video mode! ')
        rval, frame = vid.read()
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
            inference_per_frame = find_objects(className, outputs, img)
            cv2.imshow("video_stream", img)
            key = cv2.waitKey(1)
            if key == 27: # pressed escape
                cv2.destroyAllWindows()
                break
    else:
        print('image mode! ')
        rval = False
        inference_per_frame = []
        for image in os.listdir(img_path):
            
            path = os.path.join(img_path, image)
        
            if image.endswith('png'):
                img = cv2.imread(path)[...,::-1]
            else:
                img = cv2.imread(path)
                
            blob = cv2.dnn.blobFromImage(img, 1/255.0, (whT, whT), swapRB = True, crop = False)
            net.setInput(blob)
            
            layernames = net.getLayerNames()
            outputnames = [(layernames[i-1]) for i in net.getUnconnectedOutLayers()]
            # print(outputnames)
            outputs = net.forward(outputnames)
            # print(outputs)
            inference_per_frame.append(find_objects(className, outputs, img))
            cv2.imshow("a single image", img)
            key = cv2.waitKey(0) # '0' means waiting forever
            if key == ord('q'): # pressed escape
                cv2.destroyAllWindows()
    
    # df = pd.DataFrame(inference_per_frame)
    # df.to_csv('./inference_per_frame.csv', index=False)
    
    with open('./inference_per_frame.pkl', 'wb') as f:
        pickle.dump(inference_per_frame, f)


if __name__ == "__main__":
    YOLO()
   