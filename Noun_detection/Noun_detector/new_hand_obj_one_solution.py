import time

from yolo.detector_yolo import *
from new_hand_obj_detection_detector import *

import os
import numpy as np
import argparse
import cv2
import torch
import sys
sys.path.append("./lib/") 

from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
import pdb

from one_solution_efficient.model import efficientnet_b0 as create_model

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
                      default=89999, type=int) # 89999 132028
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

def load_model():
    absolute_path = os.getcwd() + '/Noun_detection/Noun_detector'
    
    args = parse_args(absolute_path)

    if args.cfg_file is not None:
      cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
      cfg_from_list(args.set_cfgs)

    cfg.USE_GPU_NMS = args.cuda
    np.random.seed(cfg.RNG_SEED)

    # load model
    model_dir = args.load_dir + "/" + args.net + "_handobj_100K" + "/" + args.dataset
    if not os.path.exists(model_dir):
      raise Exception('There is no input directory for loading network from ' + model_dir)
    load_name = os.path.join(model_dir, 'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))

    pascal_classes = np.asarray(['__background__', 'targetobject', 'hand']) 
    args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32, 64]', 'ANCHOR_RATIOS', '[0.5, 1, 2]'] 

    # initilize the network here.
    if args.net == 'vgg16':
      fasterRCNN = vgg16(pascal_classes, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res101':
      fasterRCNN = resnet(pascal_classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res50':
      fasterRCNN = resnet(pascal_classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res152':
      fasterRCNN = resnet(pascal_classes, 152, pretrained=False, class_agnostic=args.class_agnostic)
    else:
      print("network is not defined")
      pdb.set_trace()

    fasterRCNN.create_architecture()

    # print("load checkpoint %s" % (load_name))
    if args.cuda > 0:
      checkpoint = torch.load(load_name)
    else:
      checkpoint = torch.load(load_name, map_location=(lambda storage, loc: storage))
    fasterRCNN.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
      cfg.POOLING_MODE = checkpoint['pooling_mode']

    print('Load FasterRCNN successfully!')
    
    # print('image mode! ')
    return fasterRCNN, pascal_classes

def hand_obj_one_solution(image_array, model, fasterRCNN, pascal_classes):
    torch.cuda.empty_cache()
    absolute_path = os.getcwd() + '/Noun_detection/Noun_detector'

    inference_per_frame = []
    im2show, obj_name = one_soluition_hand_obj_detection_detector(image_array, inference_per_frame, absolute_path, fasterRCNN,
                                                            pascal_classes, model)

    return im2show, obj_name

            
if __name__ == "__main__":
    image_array = os.getcwd() + '/Noun_detection/Noun_detector/image'
    now = time.time()
    num_classes = 17
    model = create_model(num_classes=num_classes)
    weights_path = './one_solution_efficient/save_weights/efficientnet.h5'
    assert len(glob.glob(weights_path + "*")), "cannot find {}".format(weights_path)
    # model.load_weights(weights_path)
    model.load_weights(weights_path, by_name=True, skip_mismatch=True)
    fasterRCNN, pascal_classes = load_model()
    hand_obj_one_solution(image_array, model, fasterRCNN, pascal_classes)
