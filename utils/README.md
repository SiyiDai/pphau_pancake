# Rosbag Helper

Here we provide some instruction for prepare the recorded sequence regarding rosbag. 

## Record Rosbags with Topics

This section will instruct you how to record topic data from a running ROS system. The topic data will be accumulated in a bag file.

Before we start everything, you may want to ```mkdir``` a new directory for the recorded rosbags and the extracted images.

- **Terminal 1:**

```
roscore
```

- Connect the Realsense camera.

- **Terminal 2:**
```
roslaunch realsense2_camera rs_camera.launch
```
- **Terminal 3**

```
rviz
```

Now you can check the topic list with
```
rostopic list
```

- In rviz window, click the "Add" button in the left bottom side, can choose to create visualization "By topic".

- You may see the realtime published images from Realsense after you select the corresponding topics.


- Decide which topics you would like to record. In our case, we will need compressed image data.


```
rosbag record -j /camera/color/image_raw 
```
Here, ```-j``` means [Use BZ2 to compress data](http://wiki.ros.org/rosbag/Commandline#compress).

- If IMU data is required for recording, please check  [Realsense ros wrapper topic list](https://dev.intelrealsense.com/docs/ros-wrapper?_ga=2.29719225.767726254.1655926085-916268222.1652016004#section-published-topics) and [IMU sensor_msg documentation](http://docs.ros.org/en/lunar/api/sensor_msgs/html/msg/Imu.html) for more information.


## Extract Compressed Images from Rosbag

To extract the compressed images from a rosbag, you may first check the topics in the rosbag with ```rosbag info siyi.bag```. Once you find the compressed image topic in the rosbag, you can start to exact them by following the steps below.

- **Terminal 1:**

Since we recorded the compressed images which require us to republish to raw image format for saving, we use [image_transport](http://wiki.ros.org/image_transport) here to republish the topic.
```
rosrun image_transport republish compressed in:=camera/color/image_raw raw out:=camera_out/image
```

- **Terminal 2:**

Then we run image_saver from [image_view](http://wiki.ros.org/image_view) to save images from the new topic we republished.
```
rosrun image_view image_saver image:=/camera_out/image
```

- **Terminal 3:**
The reason for running the rosbag in the end is to ensure that we can record the whole sequence without missing the first several frames.
```
rosbag play siyi.bag
```

Now you will have all extracted images in the path of your Terminal 2, where you run image_saver.

To move all your images to another directory, you can use ```mv *.jpg /dir_name```

