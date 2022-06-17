# Object-Tracking-with-YOLACT
This is an object tracking alogorithm using yolact segmentation detector. The algorithm consists of 2 main packages : `yolact_ROS`, `mot` and those are  communicate with ROS. `yolact_ROS` is the package of detecting the position of objects and `mot` is the package of multi-object-tracking about objects and calculates velocity. To obtain position value of objects, we used depth camera model `D435i` of realsense company. Also, this algorithm only detect for person. This is overall tracking workflow.
1. Yolact find the object and its instance segmentation with object id(category number, e.g. id of person is 0)
2. Calculate center point of segmentations
3. Obtain position values(x,y,z) of center points using depth camera
4. Publish msg including `id`, `position` from `yolact_ROS` to the `mot` package
5. In `mot`, mapping the positions to the plane in one of 2 methods : `bird-eyed-view` and `forward-view` \(**Note** : For the selected method, do the same procedure 6, 7) 
6. Using multi-object-tracking algorithm in `mot` for every center points, tracking the objects and calculate the current velocity(vx,vy,vz)
7. Finally publish the `object_id`, `position`, `velocity` as a result of object tracking 

<p align="center">
  <img src = https://user-images.githubusercontent.com/78340346/173277381-7fc2c41f-d04e-47c4-9b42-9ff135cab07f.png width=640 height 480>
</p>

## Table of Content
Below is the whole procedures that we are going to do in order to do object tracking with Yolact, from start to end. The detailed explanation will be summarized in each section respectively.

A. Yolact Setups\
B. Build ROS communication system\
C. Depth Camera SDK\
D. Description of yolact_ROS\
E. Description of mot(multi-object-tracking)\
F. Launch the package\
G. Project presentation

## A. Yolact Setups
We use `Yolact` as the instance segmentation detector. While many famous real-time detectors(e.g. `Yolo`,`Faster-RCNN`) used in object tracking, but detect the large size of bounding boxes occurs much object tracking problems. As we want to detect and tracking the object more occurately, we will use `Yolact` as detector. Here are some **prerequisites of Yolact** to be installed. Also, we will use open source weight file for example `Resnet` in this object tracking section, so, if you want to train your own dataset or some other works related to Yolact, please see ![here](https://github.com/nabihandres/YOLACT/blob/main/Yolact_Tutorial.md)   
### Environment
* Ubuntu 18.04 
* NVIDIA GPU driver & CUDA 10 or later \
**Note : While installing NVIDIA, the Ubuntu default gpu driver 'nouveau' may conflic with nvidia. If that is your case, add nouveau.nomodeset=0 in grub menu.** : [Source](https://blog.neonkid.xyz/66)
* Anaconda Environment 
* PyTorch

### 1. Create a python3 based environment
Yolact operates in python3 based virtual environment, so create the virtual environment using anaconda. Referenced in source [2]
```Shell
conda create -n yolact python=3.6
```

### 2. Install Pytorch 1.0.1 or higher and TorchVision
Yolact use pytorch and torchvision which contains basic and useful deeplearning methods. Please check your pytorch version from [here](https://pytorch.org/get-started/locally/), and **find your installation command and run on your terminal**(`Run this command`) : 
 <p align="center">
  <img src = https://user-images.githubusercontent.com/78340346/157592768-b90429c1-2c6c-4e25-a5f5-4c1e41693d78.png width=640 height 480>
</p>
Then type in the terminal :

```Shell
pip3 install torch torchvision
```

### 3. Install other packages and modules
Here are some other installations of package for using yolact. (**Note** : Cython needs to be installed before pycocotools!)
```Shell
pip install cython
pip install opencv-python pillow pycocotools matplotlib
```
Setups for `Yolact` is done. Now you have to make catkin workspace and build the resources.

## B. Build ROS communication system
We will use two main packages(**yolact_ROS**, **mot**) with ROS catkin workspace. So, now we start to make catkin workspace and locate package sources. Package sources are already ready in this repository but, this sources **only include additional functions** of `yolact_ROS` and `mot` and build components. This mean you have to install `yolact` and `mot` from **github** open source espectively. Don't be confused. You can install all package resources by folling below workflow. Let's start from making catkin workspace and build.
### 1. Create workspace
```Terminal
mkdir -p tracking_ws/src
```
### 2. Build workspace
This repository contains fundamental backbones of build system(`CMakeLists.txt`, `package.xml`, `msg format` ...). So, clone `Object-Tracking-with-YOLACT` package. 
```Terminal
cd ~/tracking_ws/src
git clone https://github.com/nabihandres/YOLACT.git .
```
cakin_make to build your directory
```Terminal
cd ..
catkin_make
```
### 3. Clone Yolact and mot from github 
Now, you have to install `Yolact` and `mot` package resources from github. Please keep in mind each source files of `Yolact` and `mot` should be stored in `~/tracking_ws/src/yolact_ROS/src` and `~/tracking_ws/src/mot/src` directly. In other word, no master folder(e.g. yolact) should exist in package source directly. Therefore, clone these 2 package at your favorite workspace and just paste the sources to the package src directory. (**Note** : Clone the resources only for the purposes of using resources in open source repository. So, you can clone the yolact.git, Multi-Object-Tracking-with-Kalman-Filter.git to anywhere)
<p align="center">
  <img src = https://user-images.githubusercontent.com/78340346/172599433-ff343069-0e08-49ef-b31a-1be68b6e4a89.png width=640 height 480>
</p>

```Terminal
cd ~/ANY_DIRECTORY
git clone https://github.com/dbolya/yolact.git
# Paste resources to ~/tracking_ws/src/yolact_ROS/src
```
You need the weights file for yolact. Please download it into `yolact_ROS/src/weights` also. Download your weights from [here](https://github.com/dbolya/yolact). \
Next, please clone mot package in same way and paste resources to `~/tracking_ws/src/mot/src`. Here is mot clone command
```Terminal
cd ~/ANY_DIRECTORY
git clone https://github.com/mabhisharma/Multi-Object-Tracking-with-Kalman-Filter.git
# Paste resources to ~/tracking_ws/src/mot/src
```
## C. Depth Camera
### 1. Depth Camera SDK
In purpose of getting the position data of object, we use the depth camera product of realsense campany and model name is `D435i`. To use the depth camera you have to install `realsense SDK`. The installation page is here. Follow this [tutorial](https://github.com/IntelRealSense/librealsense/blob/master/doc/installation.md). Also you have to install pyrealsense2 library of python.
```terminal
pip install pyrealsense2
```
This library gives you much comfortable usage of depth camera using predefined pipeline. \
**Note** : The realsense-ros package is a well-known package which publish the entire data of realsense library, but the image data of ros topic published from realsense-ros should be dealed with CVBridge. And, CVBridge have many version conflict problems with python. Therefore it is proper to use pyrealsense2 library. Also, if you want to visualize the depth camera data, use realsense-viewer(Not used in this project).  \
Now you finish the entire setup of this object tracking package. Let me introduce the description of yolact and mot how they works and show you the how to Launch the package 

### 2. Coordinates
Depth camera has 3 axis of coordinates :
- x : right to the depth camera
- y : top to the depth camera
- z : forward to the depth camera

This is the coordinate image of depth camera
<p align="center">
  <img src = https://user-images.githubusercontent.com/78340346/173274072-4dd79b9c-ba63-4552-af69-43cbae4e0c28.png width=640 height 480>
</p>

## D. Description of yolact_ROS
### 1. Detect the object
Using yolact, we can detect the object and obtain the instance segmentations with object `id`. The object id is a integer number of category. For example, the person has `id` = 0.
### 2. Obtain position value
From this instance segmentation, the algorithm find the center point in frame(`pixel_x`, `pixel_y`) using concepts of `Center of Mass`. Next, using depth camera, the algorithm find the position data(`x`,`y`,`z`) from each center points in frame. Also for used in calculating velocity at mot package, time information `sec` is published.
### 3. Publish the Segment_centers msg
Every data are published in `Segment_centers` msg in `/yolact_ROS` topic. So, the msg contains `id`, `pixel_x`, `pixel_y`, `x`, `y`, `z`,`sec`.  
<p align="center">
  <img src = "https://user-images.githubusercontent.com/78340346/170453499-066a6601-f690-4bc3-9a71-6debc8962c33.png" width=640 height=480> 
</p>

### 4. Multi frame
The yolact model has the --video_multiframe argument, and by using multiframe, the yolact algorithm calculate the segmentation for parallel frames. If the yolact has the argument --video_multiframe=4, yolact calculate the 4 frames in parallel, so, it's much faster. But in this package, we only calculate in 1 frame in at the same time. So, **TODO** : the eval_ROS.py have to be added to use the --video_multiframe argument.  


## E. Description of mot(multi-object-tracking)
### 1. Problems in developed object tracking models
As I mentioned in the start of tutorial, many bounding box obtained detectors(e.g. Yolo) show errors in object tracking situations.\
There are 2 main problems in object tracking with bounding box detector : 
- Occulusion : One object may obscured by other object.
- Difficulty in Multi-object-tracking : Tracking is difficult when the objects are moving in each direction and the bounding boxes are overlapped.
### 2. Subscribe Segment-center topic
From the `yolact_ROS` package, the `mot` package subscirbe the `/yolact_ROS` topic. This is the table of `/yolact_ROS` topic variables and the used input data of object tracking algorithm.\
The description of procedure is :
- sort : Only mapping and tracking for the specific id. As the topic message has various types of objects(person, bear, chair,...), in the case of do the object traking for specific type of object, should be sorted.
- tracking : Follow the object's movement obtain the `object_id` as the result of the tracking.(**Note** : It is different from /yolact_ROS `id`.) Also, from the movement of the each objects, save the latest 10 positions in the trace and visulize on opencv.
- velocity : The object velocity in vx, vy, vz. The velocity is calculated by position data of traces.  

<p align="center">
  <img src = "https://user-images.githubusercontent.com/78340346/173303924-47d6f55a-b5ca-4f16-b63f-06f2df886ddc.png"> 
</p>  


### 3. Mapping
To improve the problems on section 1, mot do the mapping process of detected objects.\ 
There are 2 mapping methods : 
- bird-eyed-view : visualize in x-z plane (same with depth camera coordinate)
- forward-view : visualize in camera video frame (depth camera video frame)
### 4. Multi-object-tracking using kalman-filter
After mapping, I used the multi-object-tracking algorithm(source : [3]) for each mapping position data. So, In the case of using the bird-eyed-view, tracking the position data in x-z plane. In the case of using the forward-view, tracking the object center points of video frame. The tracking algorithm predict the next position of the object and if the object is located in predicted area, add the position to the trace. The trace has latest 10 object positions.\
To improve the 2 main problems mentioned in section 1, bird-eyed-view method is recommended to use. Because, after mapping in x-z plane, it is seperated from frame work problems. For more detail, If the object is obscured by other object or confused by multi objects movements, the predeveloped tracking models may confused to decide whether the new object is the same object previous frames. But, based on bird-eyed-view mapping, if the objects position is updated after few second(few frame by obscured), the previous object is located in similar position, so, distinguished in same object. 

### 5. Calculate Velocity 
After tracking the objects, using the traces of objects, the algorithm calculate the `velocity`(vx, vy, vz). The entire data(`id`, `position`, `velocity`) is published in `/object_tracking` topic. Below pictures are the result of mapping and mot. Below figure is the result of the algorithm. Green point means last 9 traces(locations) black point means the current position. 
<p align="center">
  <img src = "https://user-images.githubusercontent.com/78340346/170453504-63f05509-4b74-4953-b5af-d9297a352fd4.png" width=640 height=480> 
</p>
<p align="center">
  < bird-eyed-view tracking >
</p>
<p align="center">
  <img src = "https://user-images.githubusercontent.com/78340346/173267798-b9e0b84f-d273-423a-b09f-cbfd251a8a4b.png" width=640 height=480> \
</p>
<p align="center">
  < forward-view tracking >
</p>


## F. Launch the package
**roscore**\
Terminal 1:
```terminal
roscore
```
**yolact_ROS**\
Terminal 2:
```Terminal
cd ~/tracking_ws
source devel/setup.bash
conda activate yolact
rosrun yolact_ROS eval_ROS.py --trained_model=src/yolact_ROS/src/weights/yolact_resnet50_54_800000.pth --score_threshold=0.5 --top_k=15 --image=depth
```
**mot**\
Terminal 3:
```
cd ~/tracking_ws
source devel/setup.bash
rosrun mot object_track_ROS.py 
```

## G. Project presentation
This is the presentation video of overall project contents. The presentation deals with problems of already developed tracking algorithms and how this repository works.
<p align="center">
  <img src = "https://user-images.githubusercontent.com/78340346/172754699-c34cc034-1ddd-4533-8a56-33b314b1dce9.gif" width=640 height=480>
</p>

## Reference
[1] dbolya/yolact: A simple, fully convolutional model for real-time instance segmentation. https://github.com/dbolya/yolacthttps://github.com/dbolya/yolact, GitHub, June 9, 2022\
[2] yolact instance segmentation algorithm source code running tutorial, https://blog.katastros.com/a?ID=01200-2bc8e489-6a21-4d9f-98c3-295c7cf593eb, Katastros, June 9, 2022\
[3] mabhisharma/Multi-Object-Tracking-with-Kalman-Filter, https://github.com/mabhisharma/Multi-Object-Tracking-with-Kalman-Filter, GitHub, June 13, 2022
