# Verb Detection

#### Author: Siyi Dai

In this directory, verb detection for verb part is implemented. Instead of using images as input for detection, in this project, the hand landmarks have been used. An introduction slides is located [here](doc/Verb_Detection.pdf) with [online version](https://docs.google.com/presentation/d/1ptav9Ser1rtc2QG5Juuv3mXITvaUkN_EmHX-doMj4KY/edit?usp=sharing).

---------------

## Dependency 

**Don't forget to source to the virtual environment of pphau!!**

For this section, you will need to install the [mediapipe](https://google.github.io/mediapipe/solutions/hands) package, as well as chumpy package.


`pip install --upgrade pip`

`pip install mediapipe`

`pip install chumpy`

`pip install --upgrade tensorflow`

## File Structure

The file structure in verb_detection folder is looking like this

```
📦verb_detection
├── 📂verbs_dataset
│   ├── 📂sy0
│   ├── 📂sy1
│   └── 📂sy2
├── 📂dataset_generation
│   ├── 📜verbs_dataset_io.py
│   ├── 📜dataset_generater.py
│   ├── 📜hand_landmarks_detector.py
│   └── 📜README.md
├── 📂model
│   ├── 📜lstm_model.py
│   ├── 📜data_loader.py
│   ├── 📜data_preprocess.py
│   ├── 📜params.py
│   ├── 📜train.ipynb
│   ├── 📜time.ipynb
│   └── 📜README.md
├── 📂doc
├── 📜verb_pipeline.py
├── 📜finalized_model_5_bags_LSTM_128.sav
├── 📜finalized_model.h5
└── 📜README.md
```

---

## Main Tasks:
### Data Recording and Extraction

8 rosbags have been recorded with 4 subjects. 5 rosbags have been used for training the model with 3 subjects.

Depending on the use case, each rosbag has been extracted into images.


### Dataset Generation

As mentioned, the verb detection is based on hand landmarks instead of images. Here [mediapipe](https://google.github.io/mediapipe/solutions/hands) package has been used.


The first step is to modify mediapipe for our usage. Our aim is to save the detection into `.csv` files instead of printing in console.

Then we design the verb classes based on our story line. With consideration, we define the verbs in the story line into 12 classes, namely: 
`   "close": 0,
    "decorate": 1,
    "flip": 2,
    "move_object": 3,
    "open": 4,
    "pick_up": 5,
    "pour": 6,
    "put_down": 7,
    "screw": 8,
    "shovel": 9,
    "squeeze": 10,
    "other": 11`


We create folders for each class and we label the images by storing them into different folders. Since one verb can be appear multiple times in one record, we also label the image folders with the record and sequence names.

For each verb clip, we have one `.csv` file generated, and for each verb folder, we append the verb clips' `.csv` files.


### Data Pre-process
One drawback of mediapipe is, when there is no detection, nothing will be outputted. However, for our situation we will need a continuous input. This continuity lays both on the timestamps and features.

To solve this problem, we firstly zero-padded the missing landmarks while only one hand has been detected. Then we check the continuity in timestamp. If there is no detection in the timestamp, we zero-padded for both hand landmarks.


### Data Loader
Considering the characteristic of LSTM, we have to define a proper time step and a tride for sliding window. Multiple combinations have been tried out during development. And in the end, `timestep = 10` and `stride = 1` have been decided.

The number of features is define as hand landmarks for both hands: `21 points x (x, y, z) x (left, right)`.


### Model Building

In this project, we designed a Autoencoder-LSTM structure. After several modifications on the initial design, the model summary in the use is shown as below:
```
Model: "LSTM_128"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 0_LSTM (LSTM)               (None, 10, 128)           130560    
                                                                 
 1_Dropout (Dropout)         (None, 10, 128)           0         
                                                                 
 7_Flatten (Flatten)         (None, 1280)              0         
                                                                 
 8_Dense (Dense)             (None, 128)               163968    
                                                                 
 9_Dense (Dense)             (None, 12)                1548      
                                                                 
=================================================================
Total params: 296,076
Trainable params: 296,076
Non-trainable params: 0
_________________________________________________________________
```
In order to provide convenience for further development, all parameters have been used is saved in [params.py](model/params.py)


### Model Training
The model is trained mainly with [google collab](https://colab.research.google.com/?utm_source=scs-index). The jupyter notebook for training is saved in [trian.ipynb](model/train.ipynb)
In the end, we achieved a relatively decent training result with accuracy as **0.97** and the classification report is attached:

```
              precision    recall  f1-score   support

       close       0.96      0.77      0.85        60
    decorate       0.97      0.94      0.95       288
        flip       0.81      0.89      0.85        80
 move_object       0.78      0.75      0.77        76
        open       0.96      0.91      0.94        80
     pick_up       0.92      0.90      0.91       431
        pour       0.97      0.92      0.95       165
    put_down       0.85      0.83      0.84       209
       screw       0.93      0.98      0.95       333
      shovel       0.81      0.81      0.81       209
     squeeze       0.91      0.98      0.94       171
       other       0.99      0.99      0.99      5423

    accuracy                           0.97      7525
   macro avg       0.90      0.89      0.90      7525
weighted avg       0.97      0.97      0.97      7525


```
The current finalized model is named as [finalized_model_5_bags_LSTM_128.sav](finalized_model_5_bags_LSTM_128.sav), and also `.h5` format for compatible to other models in this project as [finalized_model.h5](finalized_model.h5).

### Verb Pipeline
To adapt the verb detection module for visualization, an verb pipeline has been developed. 

In the pipeline, we combined part of the dataset generation and data preprocess, since our input have to be as full and continuous hand landmarks and also in a length of 10 time steps.

The pipeline make use of the finalized model and predict the output, which will be a string of verb name.

The model can be tested with [verb_pipeline.py](verb_pipeline.py). Place the test images to [test](test) folder and test with one batch. 

<!-- 


### Weekly Task Update


#### 29.06 Tasks:
- Check if the presence of hands in the background causes misclassification for MP
- Continue Labeling.
- Dataloader + LSTM model.
- train/validate with 2 subjects.

#### 06.07 Updates:
- Check if the presence of hands in the background causes misclassification for MP

    - [x] checked, results in [multiple_hands](https://gitlab.lrz.de/hai-group/students/pphau/sose22/groupb/pphau_group_b/-/tree/2-verb-verb-detection-dataset-and-model-training/verb_detection/multiple_hands)
    - filter out hands: direction of the hands, vector waist to finger (Optional)

- Continue Labeling.
    - [x] one more sequence labeled, all sequences extracted
    - WRONG!!!!!! label all of the frames, if there is no verb, label it as "other"
       - frames without label, label them as "other"

- Dataloader + LSTM model.
    - [x] done, but need a good idea for data-preprocessing

- train/validate with 2 subjects.
    - [ ] not yet -->


<!-- 



## mediapipe_hands.py

This script is done following the instruction from [meidapipe](https://google.github.io/mediapipe/solutions/hands#python-solution-api). 

The idea of this script is to take a series of images and detect the hand landmarks for each image. The hand landmarks shall be saved to a folder as .json file.

---

## Questions

#### Model can be selected:
- 3dcnn
- rnn
- **lstm**  
- transformer
- vrnn...?
- social-vrnn...?


Recognition paper:
- [Pose and Joint-Aware verb Recognition](https://arxiv.org/pdf/2010.08164v2.pdf)
- [PERF-Net: Pose Empowered RGB-Flow Net](https://arxiv.org/pdf/2009.13087v2.pdf)
- [Making Convolutional Networks Recurrent for Visual Sequence Learning](https://openaccess.thecvf.com/content_cvpr_2018/papers/Yang_Making_Convolutional_Networks_CVPR_2018_paper.pdf)

Trajectory Prediction paper:
- [Social-VRNN: One-Shot Multi-modal Trajectory Prediction for Interacting Pedestrians](https://arxiv.org/pdf/2010.09056.pdf) -->