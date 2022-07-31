# Latest version in July 26th
### I have upload the classification neural network: EfficientNet model in folder: /None/Active object detection/EfficientNet_TensorFlow

### Prerequisites
- Python 3.6
- Pytorch 1.0
- CUDA 10.0
#### Create a new conda called handobj, install pytorch-1.0.1 and cuda-10.0:
```
conda create --name handobj python=3.6
conda activate handobj
conda install pytorch=1.0.1 torchvision cudatoolkit=10.0 -c pytorch
```

#### Compile the cuda dependencies using following simple commands:
```
cd lib
python setup.py build develop
```

#### Environment 
```
pip install -r requirements.txt
```

### The file structure is as followed:
```
pphau_group_b
│   README.md
│   
└───Noun_detection
|    │
|    └───Noun_detector
|    |    │
|    |    └───cropped_image
|    |    │
|    |    └───cfg 
|    |    |    │
|    |    |    └───res101.yml
|    |    |    │
|    |    |    └───res50.yml
|    |    |    │
|    |    |    └───vgg16.yml
|    |    │             
|    |    └───data
|    |    |    │
|    |    |    └───pretrained_model
|    |    |        │
|    |    |        └───resnet101_caffe.pth.txt  https://drive.google.com/file/d/1RYh8vt6cwP_bJKlcT-uNbQ1v67mQTcgp/view?usp=sharing
|    |    |
|    |    └───dataset
|    |    |     │
|    |    |     └───VGG.pth.txt  https://drive.google.com/file/d/1QwjD5mwE4F165Ii9POSnJXty5gf2Oev-/view?usp=sharing

|    |    |  
|    |    └───models/res101_handobj_100K/pascal_voc
|    |    |                                │
|    |    |                                └───faster_rcnn_1_8_132028.pth.txt  https://drive.google.com/file/d/1j0OGY0KvFUvstxfApMQ_fuJgEKBP0PN2/view?usp=sharing
|    |    |                                │
|    |    |                                └───faster_rcnn_1_8_89999.pth.txt  https://drive.google.com/file/d/1Z7lLCw9V0aPPZdZ_VHE2PFTvmpJ-8Ak3/view?usp=sharing
|    |    └───yolo/weights
|    |    |       │
|    |    |       └───yolov3_best.weights.txt  https://drive.google.com/file/d/1Ya0zRLlKZUpaBDEvuNhK6jb0dxxpF1eu/view?usp=sharing
|    |    └───image
|    |    |     │
|    |    |     └───dataset.txt  https://drive.google.com/drive/folders/1G48FpDQrJb5YEcuhJSVKbkLz3iLmad5Q?usp=sharing
|    |    └───save
|    |    |     │
|    |    |     └───**processed images**
|    |    └───lib
|    |    |     │
|    |    |     └───**needed modules**
|    |    └───yolo
|    |    |     │
|    |    |     └───**yolo modules**
|    |    │
|    |    └───new_crop_hand_obj_detection_detector.py
|    |    │
|    |    └───new_crop_hand_obj_one_solution.py
|    |    │
|    |    └───new_hand_obj_detection_detector.py
|    |    │
|    |    └───new_hand_obj_one_solution.py
|    |    │
|    |    └───VGG.py
|    |    │
|    |    └───hand_obj_detection_detector.py
|    |    │
|    |    └───hand_obj_one_solution.py
|    |    │
|    |    └───_init_paths.py
|    |    │
|    |    └───test_net.py
|    |    │
|    |    └───trainval_net.py
|    │
|    └───EfficientNet_TensorFlow
|    |    │
|    |    └───EfficientNet_pretrained_weights
|    |    |    │
|    |    |    └───EfficientNet_pretrained_weights.txt  https://drive.google.com/drive/folders/15KiWCmaMhgh3rebT-8eqOAxs_4qrQSnS?usp=sharing
|    |    │
|    |    └───data
|    |    |    │
|    |    |    └───dataset
|    |    |         │
|    |    |         └───dataset.txt  https://drive.google.com/drive/folders/1G48FpDQrJb5YEcuhJSVKbkLz3iLmad5Q?usp=sharing
|    |    |
|    |    └───save_weights
|    |    |    │
|    |    |    └───save_weights.txt  https://drive.google.com/drive/folders/1kA7yXwGqdFEIbYzFYcl8dbBLEy1LLj9f?usp=sharing
|    |    │
|    |    └───EN_test.py
|    |    │
|    |    └───EN_train.py
|    |    │
|    |    └───EfficientNet_test.ipynb
|    |    │
|    |    └───class_indices.json
|    |    │
|    |    └───model.py
|    |    │
|    |    └───predict.py
|    |    │
|    |    └───utils.py
```
### How to run Noun Detection
#### see update july 19th and prepare the environment
#### run **new_hand_obj_one_solution.py** to run the Noun Detection algorithm.
```
in Noun/Active object detection/Active_object_upload/new_hand_obj_one_solution.py
    image_array = 'C:/Users/TaoZ/TUM/HAU/hand_object_detector/image' # The place to store image to be processed

the return value of new_hand_obj_one_solution.py is the images in numpy.darray type.
```

#### run **new_crop_hand_obj_one_solution.py** to run get the cropped images from dataset.
##### the cropped images are prepared for EfficientNet
```
in Noun/Active object detection/Active_object_upload/new_crop_hand_obj_one_solution.py
    image_array = 'C:/Users/TaoZ/TUM/HAU/hand_object_detector/image' # The place to store image to be processed

the return value of new_hand_obj_one_solution.py is the cropped images and stored in file "./cropped_image".
```

#### How to train EfficientNet
##### run **EN_train.py**
```
the pretrianed weight is stored in folder "EfficientNet_pretrained_weights". you can download the ".h5" weights and put them in the folder "EfficientNet_TensorFlow"
    the pretrained weights include: 
                                    efficientnetb0.5h
                                    efficientnetb1.5h
                                    efficientnetb2.5h
                                    efficientnetb3.5h
                                    efficientnetb4.5h
                                    efficientnetb5.5h
                                    efficientnetb6.5h
                                    efficientnetb7.5h

the trained weights is stored in folder "save_weights"
    the trained weights include:
                                efficientnetB0.5h
                                efficientnetB1.5h
                                efficientnetB2.5h
                                efficientnetB3.5h
```
#### How to test EfficientNet
##### run **EfficientNet_test.ipynb**

#### The classes in EfficientNet_TensorFlow/data/dataset
```
"0": "Batter",
"1": "Batter cap",
"2": "Blueberry",
"3": "Honey",
"4": "Pan and Spatula",
"5": "Pan with Pancake (0% finished)",
"6": "Pan with Pancake (50% finished)",
"7": "Pan with Pancake (80% finished)",
"8": "Pan_handle",
"9": "Pancake",
"10": "Plate",
"11": "Raspberry",
"12": "Rest",
"13": "Spatula",
"14": "Spatula with Pancake (100% finished)",
"15": "Spatula with Pancake (50% finished)",
"16": "Spatula with Pancake (80% finished)"
```


# update july 19th
### I have upload the Active object detection model in folder: /None/Active object detection/Active_object_upload

### There are several big documents that storaged in goolg drive
```
pphau_group_b
│   README.md
│
└───Noun/Active object detection
    │
    └───Active_object_upload
            │
            └───data
            |    │
            |    └───pretrained_model
            |        │
            |        └───resnet101_caffe.pth.txt
            |
            └───dataset
            |     │
            |     └───VGG.pth.txt
            |  
            └───models/res101_handobj_100K/pascal_voc
            |                                │
            |                                └───faster_rcnn_1_8_132028.pth.txt
            └───yolo/weights
                        │
                        └───yolov3_best.weights.txt
```
### Prerequisites
- Python 3.6
- Pytorch 1.0
- CUDA 10.0
#### Create a new conda called handobj, install pytorch-1.0.1 and cuda-10.0:
```
conda create --name handobj python=3.6
conda activate handobj
conda install pytorch=1.0.1 torchvision cudatoolkit=10.0 -c pytorch
```

#### Compile the cuda dependencies using following simple commands:
```
cd lib
python setup.py build develop
```

#### Environment 
```
pip install -r requirements.txt
```
#### Execution
- run hand_obj_one_solution.py

please change the **PATH** in **hand_obj_one_solution.py**

```
if __name__ == "__main__":
    PATH = 'C:/Users/TaoZ/TUM/HAU/hand_obj_detector_upload'
    img_path = "image"

```

# update july 5th
## How to do Noun/Active object detection?

> ### The features of an active object
> - The active object has interaction with hands
> - The active object is the only object that moves in serveral continuous frame in a video with hands

> ### Based on these features of active objects, I propose two methods to complete this subporject

> ##### Using color image and depth image
> 1. Do active object detection with hand detection and depth information
> - since active object moves with hands. We can firstly use hand detection method to localise hand in a picture and then get its depth. At the same time, we can extract several candidate objects that have interrelation with hand in the picture. Finally, we can use the depth information to localise the active object by comparing it with the hand's depth information. 

> ##### Using color image only
> 1. Active object detection is also an emerging research filed in computer vision and there are a lot of outstanding works to do such a detection only via color images.
> 2. In this subproject, I also want to explore the state-of-art algorithm to do active object detection
> 3. There an algorithm which is propsed by researchers from Carnegie Mellon University, its name is 
_Sequential Voting with Relational Box Fields for Active Object Detection_
>> - In this paper, __they propose a pixelwise voting function. Their pixel-wise voting function takes an initial bounding box as input and produces an improved bounding box of the active object as output.__
>> - It only uses the color information and achieves a relative high detection accuracy.
>> - What's more, the authors release their work on github and I can implement this method directly on our database.
>> - <https://fuqichen1998.github.io/SequentialVotingDet/.>
