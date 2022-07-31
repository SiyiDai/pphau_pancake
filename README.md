# pphau_group_B: Perfect Pancake 🥞

The goal of this project is to learn and implement the key components required for a vision-based Human Activity Understanding pipeline.



## Project Introduction
**Input:**

Images <-- Rosbags


**Output:**

Objects + Positions <-- [Object_detection](Object_detection)

Verb + Noun <-- [Verb_detection](Verb_detection) + [Noun_detection](Noun_detection)


**Ego-Perspective:**

- Head mounted RGB-D camera 
- Ego-Perspective data

**Problem Domain:**

Kitchen: 

To make a “perfect” pancake from zero
 
## Story Line

Check here to see the [Story Line](https://gitlab.lrz.de/hai-group/students/pphau/sose22/groupb/pphau_group_b/-/wikis/Story%20Line)

## Project Structure
![1](https://raw.githubusercontent.com/SiyiDai/pphau_pancake/main/Final%20Presentation/overall.png)


### Pipeline Overview
- Data Collection with Randomizations
    - 20 rosbags with 4 subjects, recorded 3 times 
- Semantic Information Extraction
    - Rosbags → Images
- Recognize Activities
    - [Object Detection](Object_detection) → Shuang Wang
    - [Noun Detection](Noun_detection) → Zheng Tao
    - [Verb Detection](Verb_detection) → Siyi Dai
- Update Models
    - YOLO, Efficient Net, LSTM
- Visualization
    - [Visualizer](visualizer.py) → Chi Wang

### Run Visualizer
`roscore`


`python3 visualizer.py`


`rosbag play -l example.bag`

