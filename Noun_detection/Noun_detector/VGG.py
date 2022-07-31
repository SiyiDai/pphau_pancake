import torch
from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy


def VGG():
    os.chdir('C:/Users/TaoZ/TUM/HAU/hand_object_detector/dataset/')
    
    
    transform = transforms.Compose([            #[1]
     transforms.Resize(256),                    #[2]
     transforms.CenterCrop(224),                #[3]
     transforms.ToTensor(),                     #[4]
     transforms.Normalize(                      #[5]
     mean=[0.485, 0.456, 0.406],                #[6]
     std=[0.229, 0.224, 0.225]                  #[7]
     )])
    
    num_classes = 5
    model_VGG = models.vgg16()
    num_ftrs = model_VGG.classifier[6].in_features
    model_VGG.classifier[6] = nn.Linear(num_ftrs, num_classes)
    
    device = torch.device('cpu')
    
    model_VGG.load_state_dict(torch.load('VGG.pth', map_location=device))
    # print(torch.cuda.is_available())
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_VGG.eval().to(device)
    
    path = 'test/left1200cw.jpg'
    
    # Read the image
    image = Image.open(path)
    
    # Convert the image to PyTorch tensor
    image_tensor = transform(image)
    
    batch_t = torch.unsqueeze(image_tensor, 0)
    
    
    predicted = model_VGG(batch_t)
    
    with open('./image_classes.txt') as f:
      classes = [line.strip() for line in f.readlines()]
    
    _, index = torch.max(predicted, 1)
    
    percentage = torch.nn.functional.softmax(predicted, dim=1)[0] * 100
    
    print(classes[index[0]], percentage[index[0]].item())
    
    print(classes[index[0]])
    
    
def VGG_imread(image, PATH):
    # os.chdir('C:/Users/TaoZ/TUM/HAU/hand_object_detector/dataset/')
    VGG_path = os.path.join(PATH, 'dataset/VGG.pth')
    image_class_path = os.path.join(PATH, 'dataset/image_classes.txt')
    
    
    transform = transforms.Compose([            #[1]
     transforms.Resize(256),                    #[2]
     transforms.CenterCrop(224),                #[3]
     transforms.ToTensor(),                     #[4]
     transforms.Normalize(                      #[5]
     mean=[0.485, 0.456, 0.406],                #[6]
     std=[0.229, 0.224, 0.225]                  #[7]
     )])
    
    num_classes = 5
    model_VGG = models.vgg16()
    num_ftrs = model_VGG.classifier[6].in_features
    model_VGG.classifier[6] = nn.Linear(num_ftrs, num_classes)
    
    device = torch.device('cpu')
    
    model_VGG.load_state_dict(torch.load(VGG_path, map_location=device))
    # print(torch.cuda.is_available())
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_VGG.eval().to(device)
    
    # Convert the image to PyTorch tensor
    image_tensor = transform(image)
    
    batch_t = torch.unsqueeze(image_tensor, 0)
    
    
    predicted = model_VGG(batch_t)
    
    with open(image_class_path) as f:
      classes = [line.strip() for line in f.readlines()]
    
    _, index = torch.max(predicted, 1)
    
    percentage = torch.nn.functional.softmax(predicted, dim=1)[0] * 100
    
    class_name = classes[index[0]]
    
    class_percentage =  percentage[index[0]].item()
    
    # print(class_name, class_percentage)
    
    return class_name

if __name__ == "__main__":
    VGG()

