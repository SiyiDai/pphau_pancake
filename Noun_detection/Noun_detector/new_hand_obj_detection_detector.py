from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
from PIL import Image

import torchvision.transforms as transforms
import torchvision.datasets as dset
# from scipy.misc import imread
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
# from model.nms.nms_wrapper import nms
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections, vis_detections_PIL, vis_detections_filtered_objects_PIL_classification, vis_detections_filtered_objects, calculate_center # (1) here add a function to viz
from model.utils.blob import im_list_to_blob
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
import pdb

import pandas as pd
import pickle

import math

from PIL import Image, ImageDraw, ImageFont
from model.utils.viz_hand_obj import *

# try:
#     xrange          # Python 2
# except NameError:
#     xrange = range  # Python 3
    
    
###############################################################################
def calculate_cc(bbox):
    x_c = (bbox[0] + bbox[2])/2.0
    y_c = (bbox[1] + bbox[3])/2.0
    
    return x_c, y_c
    
def calculate_distance(bbox1, bbox2):
    x1_c, y1_c = calculate_cc(bbox1)
    x2_c, y2_c = calculate_cc(bbox2)
    
   
    distance = math.sqrt( (x2_c - x1_c)**2 + (y2_c - y1_c)**2 )
    
    return distance

def spatula_and_pancake(bbox_s, bbox_p):
    threshold = 50
    
    x_s, y_s = calculate_cc(bbox_s)
    x_p, y_p = calculate_cc(bbox_p)
    
    distance = math.sqrt( (x_s - x_p)**2 + (y_s - y_p)**2 )
    
    if distance < threshold:
        return True
    else:
        return False

def calculate_IoU(bbox, loaded_dict):
    iou_flag = 0
    threshold_dis = 150
    
    obj_name = 'other'
    
    iou_all = []
    name = []
    
    target_bbox = bbox
    
    for item in loaded_dict.keys():
        obj = loaded_dict[item]['bbox']
        
        x_obj_1 = obj[0, 0]
        y_obj_1 = obj[0, 1]
        x_obj_2 = obj[0, 0] + obj[1, 0]
        y_obj_2 = obj[0, 1] + obj[1, 1]
        
        obj_bbox = [x_obj_1, y_obj_1, x_obj_2, y_obj_2]
        
        xA = max(target_bbox[0], obj_bbox[0])
        yA = max(target_bbox[1], obj_bbox[1])
        xB = min(target_bbox[2], obj_bbox[2])
        yB = min(target_bbox[3], obj_bbox[3])
    
        # compute the area of intersection rectangle
        interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
        # if interArea == 0:
        #     return 0
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = abs((target_bbox[2] - target_bbox[0]) * (target_bbox[3] - target_bbox[1]))
        boxBArea = abs((obj_bbox[2] - obj_bbox[0]) * (obj_bbox[3] - obj_bbox[1]))
    
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
        iou_all.append(iou)
        name.append(item)
        if iou > iou_flag and iou > 0.50:
            distance = calculate_distance(target_bbox, obj_bbox)
            if distance < threshold_dis:
                iou_flag = iou
                obj_name = item
                
                # print("iou_flag", iou_flag)
                # print('distance', distance)
        # else:
        #     print('yolo cannot find proper catagory')
    return obj_name

def coordinate_map_from_yolo(bbox):
    x_obj_1 = bbox[0, 0]
    y_obj_1 = bbox[0, 1]
    x_obj_2 = bbox[0, 0] + bbox[1, 0]
    y_obj_2 = bbox[0, 1] + bbox[1, 1]
    
    bbox_p = np.zeros((1, 10))
    bbox_p[0, 0:4] = [x_obj_1, y_obj_1, x_obj_2, y_obj_2]
    return bbox_p
###############################################################################


def parse_args(PATH):
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='pascal_voc', type=str)
  parser.add_argument('--cfg', dest='cfg_file',
                      help='optional config file',
                      default=os.path.join(PATH, 'cfgs/res101.yml'), type=str)
  parser.add_argument('--net', dest='net',
                      help='vgg16, res50, res101, res152',
                      default='res101', type=str)
  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)
  parser.add_argument('--load_dir', dest='load_dir',
                      help='directory to load models',
                      default=os.path.join(PATH, "models"))
  parser.add_argument('--image_dir', dest='image_dir',
                      help='directory to load images for demo',
                      default=os.path.join(PATH, 'image')) # C:/Users/TaoZ/TUM/HAU/hand_object_detector
  parser.add_argument('--save_dir', dest='save_dir',
                      help='directory to save results',
                      default=os.path.join(PATH, 'save')) # C:/Users/TaoZ/TUM/HAU/hand_object_detector
  parser.add_argument('--cuda', dest='cuda', 
                      default= True,
                      help='whether use CUDA',
                      action='store_true')
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether perform class_agnostic bbox regression',
                      action='store_true')
  parser.add_argument('--parallel_type', dest='parallel_type',
                      help='which part of model to parallel, 0: all, 1: model before roi pooling',
                      default=0, type=int)
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load network',
                      default=8, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load network',
                      default=132028, type=int)
  parser.add_argument('--bs', dest='batch_size',
                      help='batch_size',
                      default=1, type=int)
  parser.add_argument('--vis', dest='vis',
                      help='visualization mode',
                      default=True)
  parser.add_argument('--webcam_num', dest='webcam_num',
                      help='webcam ID number',
                      default=-1, type=int)
  parser.add_argument('--thresh_hand',
                      type=float, default=0.5,
                      required=False)
  parser.add_argument('--thresh_obj', default=0.5,
                      type=float,
                      required=False)

  args = parser.parse_args()
  return args

lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

def _get_image_blob(im):
  """Converts an image into a network input.
  Arguments:
    im (ndarray): a color image in BGR order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
  im_orig = im.astype(np.float32, copy=True)
  im_orig -= cfg.PIXEL_MEANS

  im_shape = im_orig.shape
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])

  processed_ims = []
  im_scale_factors = []

  for target_size in cfg.TEST.SCALES:
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
      im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
            interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, np.array(im_scale_factors)


          
          
def one_soluition_hand_obj_detection_detector(image, loaded_dict, absolute_path, fasterRCNN, pascal_classes, model):
    args = parse_args(absolute_path)
    
    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)
    box_info = torch.FloatTensor(1) 

    # ship to cuda
    if args.cuda > 0:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    with torch.no_grad():
        if args.cuda > 0:
            cfg.CUDA = True
    
        if args.cuda > 0:
            print("Faster RCNN is Using CUDA")
            fasterRCNN.cuda()

        fasterRCNN.eval()
      
        start = time.time()
        max_per_image = 100
        thresh_hand = args.thresh_hand 
        thresh_obj = args.thresh_obj
        vis = args.vis
      
        webcam_num = args.webcam_num
      
        total_tic = time.time()
       
        im_in = image
        # bgr
        im = im_in
      
        blobs, im_scales = _get_image_blob(im)
        assert len(im_scales) == 1, "Only single-image batch implemented"
        im_blob = blobs
        im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)
      
        im_data_pt = torch.from_numpy(im_blob)
        im_data_pt = im_data_pt.permute(0, 3, 1, 2)
        im_info_pt = torch.from_numpy(im_info_np)
      
        with torch.no_grad():
            im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
            im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
            gt_boxes.resize_(1, 1, 5).zero_()
            num_boxes.resize_(1).zero_()
            box_info.resize_(1, 1, 5).zero_() 

      # pdb.set_trace()
        det_tic = time.time()
    
        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label, loss_list = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, box_info) 
    
        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]
    
        # extact predicted params
        contact_vector = loss_list[0][0] # hand contact state info
        offset_vector = loss_list[1][0].detach() # offset vector (factored into a unit vector and a magnitude)
        lr_vector = loss_list[2][0].detach() # hand side info (left/right)
    
        # get hand contact 
        _, contact_indices = torch.max(contact_vector, 2)
        contact_indices = contact_indices.squeeze(0).unsqueeze(-1).float()
    
        # get hand side 
        lr = torch.sigmoid(lr_vector) > 0.5
        lr = lr.squeeze(0).float()
    
        if cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred.data
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            # Optionally normalize targets by a precomputed mean and stdev
              if args.class_agnostic:
                  if args.cuda > 0:
                      box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                  else:
                      box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
    
                  box_deltas = box_deltas.view(1, -1, 4)
              else:
                  if args.cuda > 0:
                      box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                  else:
                      box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
                  box_deltas = box_deltas.view(1, -1, 4 * len(pascal_classes))
    
            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))

        pred_boxes /= im_scales[0]
    
        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()
        det_toc = time.time()
        detect_time = det_toc - det_tic
        print("Process FasterRCNN(s):", detect_time)
        misc_tic = time.time()
        if vis:
            im2show = np.copy(im)
        obj_dets, hand_dets = None, None
        for j in range(1, len(pascal_classes)):
            # inds = torch.nonzero(scores[:,j] > thresh).view(-1)
            if pascal_classes[j] == 'hand':
              inds = torch.nonzero(scores[:,j]>thresh_hand).view(-1)
            elif pascal_classes[j] == 'targetobject':
              inds = torch.nonzero(scores[:,j]>thresh_obj).view(-1)
    
            # if there is det
            if inds.numel() > 0:
              cls_scores = scores[:,j][inds]
              _, order = torch.sort(cls_scores, 0, True)
              if args.class_agnostic:
                cls_boxes = pred_boxes[inds, :]
              else:
                cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
              
              cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1), contact_indices[inds], offset_vector.squeeze(0)[inds], lr[inds]), 1)
              cls_dets = cls_dets[order]
              keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
              cls_dets = cls_dets[keep.view(-1).long()]
              if pascal_classes[j] == 'targetobject':
                obj_dets = cls_dets.cpu().numpy()
              if pascal_classes[j] == 'hand':
                hand_dets = cls_dets.cpu().numpy()

###############################################################################
        # draw the original BBOX found by hand_obj_detector
        # convert to PIL
        im_PIL = image.copy()
        im_PIL = im_PIL[:,:,::-1]
        im_PIL = Image.fromarray(im_PIL).convert("RGB")
        draw = ImageDraw.Draw(im_PIL)
        # font = ImageFont.truetype('./lib/model/utils/times_b.ttf', size=30)
        width, height = im_PIL.size
        if obj_dets is not None:
             for i in range(len(obj_dets)):
                 obj_bbox = obj_dets[i, :4]
                 im_PIL = draw_obj_mask_simpler(im_PIL, draw, obj_bbox, width, height)
                
        # im_PIL = np.array(im_PIL)
        # cv2.imshow("a single image", im_PIL)
        # # if cv2.waitKey(1): #
        # #     cv2.destroyAllWindows()
         # exit(100)
###############################################################################
        if obj_dets is not None:

        #   for item in loaded_dict.keys():
        #       if item.startswith('Pancake'):
        #           obj = loaded_dict[item]['bbox']
        #           bbox_p = coordinate_map_from_yolo(obj)
        #           obj_dets = np.append(obj_dets, bbox_p, axis=0)
        #           break

###############################################################################
            if vis:
                obj_name = []
            # visualization
            # using EfficientNet as an auxilary branch
                im2show, obj_name= \
                    vis_detections_filtered_objects_PIL_classification(im2show, obj_dets, hand_dets, absolute_path, model, thresh_hand, thresh_obj)

            misc_toc = time.time()
            nms_time = misc_toc - misc_tic
  
            if webcam_num == -1:
                sys.stdout.write('im_detect: {:.3f}s {:.3f}s   \r' \
                                .format(detect_time, nms_time))
                sys.stdout.flush()
  
            if vis and webcam_num == -1:
###############################################################################
              im2show = np.array(im2show)
              im2show = im2show[:, :, ::-1].copy()
###############################################################################             
            else:
              im2showRGB = cv2.cvtColor(im2show, cv2.COLOR_BGR2RGB)
              cv2.imshow("frame", im2showRGB)
              total_toc = time.time()
              total_time = total_toc - total_tic
              frame_rate = 1 / total_time
              print('Frame rate:', frame_rate)
              # if cv2.waitKey(1) & 0xFF == ord('q'):
              #     break

          # print("the active objects are:", obj_name)
            return im2show, obj_name
      
        else:
            print('there is no hand in image')
            image_without_text = copy.deepcopy(im2show)
            obj_name = []
            return im2show, obj_name


# if __name__ == '__main__':
#     hand_obj_detection_detector()
    
    
    
    
# def hand_obj_detection_detector():
#     args = parse_args()

#     # print('Called with args:')
#     # print(args)

#     if args.cfg_file is not None:
#       cfg_from_file(args.cfg_file)
#     if args.set_cfgs is not None:
#       cfg_from_list(args.set_cfgs)

#     cfg.USE_GPU_NMS = args.cuda
#     np.random.seed(cfg.RNG_SEED)

#     # load model
#     model_dir = args.load_dir + "/" + args.net + "_handobj_100K" + "/" + args.dataset
#     if not os.path.exists(model_dir):
#       raise Exception('There is no input directory for loading network from ' + model_dir)
#     load_name = os.path.join(model_dir, 'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))

#     pascal_classes = np.asarray(['__background__', 'targetobject', 'hand']) 
#     args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32, 64]', 'ANCHOR_RATIOS', '[0.5, 1, 2]'] 

#     # initilize the network here.
#     if args.net == 'vgg16':
#       fasterRCNN = vgg16(pascal_classes, pretrained=False, class_agnostic=args.class_agnostic)
#     elif args.net == 'res101':
#       fasterRCNN = resnet(pascal_classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
#     elif args.net == 'res50':
#       fasterRCNN = resnet(pascal_classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
#     elif args.net == 'res152':
#       fasterRCNN = resnet(pascal_classes, 152, pretrained=False, class_agnostic=args.class_agnostic)
#     else:
#       print("network is not defined")
#       pdb.set_trace()

#     fasterRCNN.create_architecture()

#     print("load checkpoint %s" % (load_name))
#     if args.cuda > 0:
#       checkpoint = torch.load(load_name)
#     else:
#       checkpoint = torch.load(load_name, map_location=(lambda storage, loc: storage))
#     fasterRCNN.load_state_dict(checkpoint['model'])
#     if 'pooling_mode' in checkpoint.keys():
#       cfg.POOLING_MODE = checkpoint['pooling_mode']

#     print('load model successfully!')


#     # initilize the tensor holder here.
#     im_data = torch.FloatTensor(1)
#     im_info = torch.FloatTensor(1)
#     num_boxes = torch.LongTensor(1)
#     gt_boxes = torch.FloatTensor(1)
#     box_info = torch.FloatTensor(1) 

#     # ship to cuda
#     if args.cuda > 0:
#       im_data = im_data.cuda()
#       im_info = im_info.cuda()
#       num_boxes = num_boxes.cuda()
#       gt_boxes = gt_boxes.cuda()

#     with torch.no_grad():
#       if args.cuda > 0:
#         cfg.CUDA = True

#       if args.cuda > 0:
#         fasterRCNN.cuda()

#       fasterRCNN.eval()

#       start = time.time()
#       max_per_image = 100
#       thresh_hand = args.thresh_hand 
#       thresh_obj = args.thresh_obj
#       vis = args.vis

#       # print(f'thresh_hand = {thresh_hand}')
#       # print(f'thnres_obj = {thresh_obj}')

#       webcam_num = args.webcam_num
#       # Set up webcam or get image directories
#       if webcam_num >= 0 :
#         cap = cv2.VideoCapture(webcam_num)
#         num_images = 0
#       else:
#         print(f'image dir = {args.image_dir}')
#         print(f'save dir = {args.save_dir}')
#         imglist = os.listdir(args.image_dir)
#         num_images = len(imglist)

#       print('Loaded Photo: {} images.'.format(num_images))
      
#       inference_path = './yolo/inference_per_frame.pkl'
#       with open(inference_path, 'rb') as f:
#           loaded_dict = pickle.load(f)
#       # keys = loaded_dict.keys()

#       while (num_images >= 0):
#           total_tic = time.time()
#           if webcam_num == -1:
#             num_images -= 1

#           # Get image from the webcam
#           if webcam_num >= 0:
#             if not cap.isOpened():
#               raise RuntimeError("Webcam could not open. Please check connection.")
#             ret, frame = cap.read()
#             im_in = np.array(frame)
#           # Load the demo image
#           else:
#             im_file = os.path.join(args.image_dir, imglist[num_images])
#             im_in = cv2.imread(im_file)
#           # bgr
#           im = im_in

#           blobs, im_scales = _get_image_blob(im)
#           assert len(im_scales) == 1, "Only single-image batch implemented"
#           im_blob = blobs
#           im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

#           im_data_pt = torch.from_numpy(im_blob)
#           im_data_pt = im_data_pt.permute(0, 3, 1, 2)
#           im_info_pt = torch.from_numpy(im_info_np)

#           with torch.no_grad():
#                   im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
#                   im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
#                   gt_boxes.resize_(1, 1, 5).zero_()
#                   num_boxes.resize_(1).zero_()
#                   box_info.resize_(1, 1, 5).zero_() 

#           # pdb.set_trace()
#           det_tic = time.time()

#           rois, cls_prob, bbox_pred, \
#           rpn_loss_cls, rpn_loss_box, \
#           RCNN_loss_cls, RCNN_loss_bbox, \
#           rois_label, loss_list = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, box_info) 

#           scores = cls_prob.data
#           boxes = rois.data[:, :, 1:5]

#           # extact predicted params
#           contact_vector = loss_list[0][0] # hand contact state info
#           offset_vector = loss_list[1][0].detach() # offset vector (factored into a unit vector and a magnitude)
#           lr_vector = loss_list[2][0].detach() # hand side info (left/right)

#           # get hand contact 
#           _, contact_indices = torch.max(contact_vector, 2)
#           contact_indices = contact_indices.squeeze(0).unsqueeze(-1).float()

#           # get hand side 
#           lr = torch.sigmoid(lr_vector) > 0.5
#           lr = lr.squeeze(0).float()

#           if cfg.TEST.BBOX_REG:
#               # Apply bounding-box regression deltas
#               box_deltas = bbox_pred.data
#               if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
#               # Optionally normalize targets by a precomputed mean and stdev
#                 if args.class_agnostic:
#                     if args.cuda > 0:
#                         box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
#                                   + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
#                     else:
#                         box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
#                                   + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)

#                     box_deltas = box_deltas.view(1, -1, 4)
#                 else:
#                     if args.cuda > 0:
#                         box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
#                                   + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
#                     else:
#                         box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
#                                   + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
#                     box_deltas = box_deltas.view(1, -1, 4 * len(pascal_classes))

#               pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
#               pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
#           else:
#               # Simply repeat the boxes, once for each class
#               pred_boxes = np.tile(boxes, (1, scores.shape[1]))

#           pred_boxes /= im_scales[0]

#           scores = scores.squeeze()
#           pred_boxes = pred_boxes.squeeze()
#           det_toc = time.time()
#           detect_time = det_toc - det_tic
#           misc_tic = time.time()
#           if vis:
#               im2show = np.copy(im)
#           obj_dets, hand_dets = None, None
#           for j in xrange(1, len(pascal_classes)):
#               # inds = torch.nonzero(scores[:,j] > thresh).view(-1)
#               if pascal_classes[j] == 'hand':
#                 inds = torch.nonzero(scores[:,j]>thresh_hand).view(-1)
#               elif pascal_classes[j] == 'targetobject':
#                 inds = torch.nonzero(scores[:,j]>thresh_obj).view(-1)

#               # if there is det
#               if inds.numel() > 0:
#                 cls_scores = scores[:,j][inds]
#                 _, order = torch.sort(cls_scores, 0, True)
#                 if args.class_agnostic:
#                   cls_boxes = pred_boxes[inds, :]
#                 else:
#                   cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
                
#                 cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1), contact_indices[inds], offset_vector.squeeze(0)[inds], lr[inds]), 1)
#                 cls_dets = cls_dets[order]
#                 keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
#                 cls_dets = cls_dets[keep.view(-1).long()]
#                 if pascal_classes[j] == 'targetobject':
#                   obj_dets = cls_dets.cpu().numpy()
#                 if pascal_classes[j] == 'hand':
#                   hand_dets = cls_dets.cpu().numpy()

# ###############################################################################
#           # print('\n image_name:', imglist[num_images])
#           # print('\n obj_dets:', obj_dets[0, :4])
#           # print('\n hand_dets:', hand_dets)
#           name_append = None
#           if obj_dets is not None:
#               obj_name = []
#               for i in range(len(obj_dets)):
#                   bbox = obj_dets[i, :4]
                  
#                   obj_name.append(calculate_IoU(bbox, loaded_dict[num_images]))
                  
#               if 'spatula' in obj_name:
#                   for item in loaded_dict[num_images].keys():
#                       if item.startswith('pancake'):
#                           ind = obj_name.index('spatula')
#                           bbox_s = obj_dets[ind, :4]
#                           obj = loaded_dict[item]['bbox']
#                           bbox_p = [obj[0, 0], obj[0, 1], obj[0, 0] + obj[1, 0], obj[0, 1] + obj[1, 1]]
#                           if spatula_and_pancake(bbox_s, bbox_p):
#                               name_append = item
#                               break

# ###############################################################################
#               if vis:
#                 # visualization
#                 im2show = vis_detections_filtered_objects_PIL(im2show, obj_dets, hand_dets, obj_name, thresh_hand, thresh_obj)

#               misc_toc = time.time()
#               nms_time = misc_toc - misc_tic

#               if webcam_num == -1:
#                   sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
#                                   .format(num_images + 1, len(imglist), detect_time, nms_time))
#                   sys.stdout.flush()
      
#               if vis and webcam_num == -1:
# ###############################################################################
#                   im2show = np.array(im2show)
#                   im2show = im2show[:, :, ::-1].copy()
#                   # im2showRGB = cv2.cvtColor(im2show, cv2.COLOR_BGR2RGB)
#                   # print(type(im2show))
#                   cv2.imshow('image', im2show)
#                   key = cv2.waitKey(0) # '0' means waiting forever
#                   if key == ord('q'): # pressed escape
#                       cv2.destroyAllWindows()
                  
#                   im2show = im2show[:,:,::-1]
#                   im2show = Image.fromarray(im2show).convert("RGBA")
# ###############################################################################
#                   folder_name = args.save_dir
#                   os.makedirs(folder_name, exist_ok=True)
#                   result_path = os.path.join(folder_name, imglist[num_images][:-4] + "_det.png")
#                   im2show.save(result_path)
#               else:
#                   im2showRGB = cv2.cvtColor(im2show, cv2.COLOR_BGR2RGB)
#                   cv2.imshow("frame", im2showRGB)
#                   total_toc = time.time()
#                   total_time = total_toc - total_tic
#                   frame_rate = 1 / total_time
#                   print('Frame rate:', frame_rate)
#                   if cv2.waitKey(1) & 0xFF == ord('q'):
#                       break
#               if name_append is not None:
#                   obj_name.append(name_append)
#                   print("the active objects are:", obj_name)
#               else:
#                   print("the active objects are:", obj_name)
#           else:
#               print('there is no hand in image:', im_file)
              
#       if webcam_num >= 0:
#           cap.release()
#           cv2.destroyAllWindows()