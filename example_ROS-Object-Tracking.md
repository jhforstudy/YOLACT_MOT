# ROS-Object-Tracking
This repositorycontains ROS package `object_tracking` to track the object and calculate its velocity. By using object detection, we obtain our target object bounding box in the topic'/darknet_ros/bounding_boxes'. Then, we find the center point of bounding box, obtain depth from '/camera/depth/image_rect_raw' to start tracking the center of the boundig box and calculates its postition and velocity.

Note that this is a ROS package. Please make sure you have ROS melodic in your UBUNTU 18.04 operating system.\
**By using this repository, the velocity of target object can be computed by implemmenting object tracking algorithm.\
The output velocity of the tracking point (red dot) is in millimeter(mm) as show in the figure 1.

![image](example.png)\
Figure 1. tracking point of the bounding box (center point).

**The output of code is decribed here:**
>x, y, z = local-X, local-Y, local-Z positions (mm) of target object in respect to camera frame.\
>vx, vy, vz = local velocity-X, local velocity-Y, local velocity-Z (mm/s) of target object in respect to camera frame.\
>velocity = velocity of target object (mm/s) in the vector direction.

**Please look at below explanation for more details.**

## 0. Concept Explanation
### Introduction
![fov](/fov.png)\
Figure 2. Coordinate frame of the Realsense D435i
[Reference](https://www.intel.com/content/dam/support/us/en/documents/emerging-technologies/intel-realsense-technology/Intel-RealSense-D400-Series-Datasheet.pdf)\
There are several terms that we need to know. Horizontal field of vision (HFOV) and vertical field of vision (VFOV), which showed at the figure above.\
Positive Z-axis --> Moving away from camera\
Positive Y-axis --> Moving up in respect to camera\
Positive X-axis --> Moving right in respect to camera\
\
These axes are all relative to local frame of camera.

### Procedure for Object Tracking Algorithm
[Reference](https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/)\
**This is just some brief explanations regarding the implementation procedure for object tracking, but it does cover the core concepts.**\
\
First, Create a dictionary ( ex: **Objects**={ ID : information } ) for storing detected objects with their information. 
For each frame of the video, we get object location of centre point (x, y) of bounding box. then do as below:
1. Get the x, y, and depth of each centre point of bounding boxes
2. Compute the local position of the point in reference to camera. (convert pixel position to mm )
3. **If** dict **Objects**.size = 0 it means there is not any bounding box founded, assign new id to this object with corresponding information into **Objects**.These information included:\
	i. x and y of centre point\
	ii. local position (mm) of centre point respect to camera\
	iii. class\
	iv. The time it appears\
	v. distance it travelled in x y z **calc_dist** function  final pos - initial pos
	(of course will be set to 0 in the first time it appears!)\
	
	**Else if** the Object.size != 0, we use **update** function to find if this object is actually the same as one of the object from **Objects**, by analyzing its x, y position of centre point of bounding box. \
	**If** the object found to be the same object from **Objects**, then update its information, such as\
	i. x and y of centre point\
	ii. local position of centre point\
	iii. distance it travelled( we can compute this from its previous local position and current local position)\
	iv. local velocity of the object in respect to camera, by using the **calc_vel** function.\
	we set value (10) in the maxFrameCount variable that means we are going calculate the velocity of the object after it appear 10 times continuosly in **update** function
	**Else** the object is not from **Objects**, do step 3.
	
4. We update the **Objects** dictionary, then using the **draw** function we print all the objects informations (bounding boxes) to the screen. 
Note: If the object is **traffic light**, we will use **colour_detector** function to detect the colour of traffic light, and draw the result on the screen.\

## 1. Installation
We need to download and put this **object_tracking** package with the **darknet_ros** package inside to **object_detection_ws** (workspace).  
```
cd ~/catkin_workspace/object_detection_ws/src
git clone https://github.com/CharmFlex-98/ROS-Object-Tracking.git
cd ..
catkin_make --pkg object_tracking
```
Make sure the required dependencies are installed in order to run this package successfully.
```
rosdep install --from-paths ./src --ignore-src --rosdistro=melodic
```
## 2. Run the Package
******************************************************
**If this is your first time running this package, please edit the ros.yaml from the `darknet_ros` folder**\
Since our script and `darknet_ros` both will display window for results object tracking and object detection, we will combine them into one window.\
We need disable `opencv` in `darknet_ros`
```
cd nano ~/catkin_workspace/object_detection_ws/src/darknet_ros/darknet_ros/config/ros.yaml
```
Under `image_view`, set:
```
enable_opencv: false
```
and save:
```
ctrl x -> y -> enter
```
******************************************************
In order to run the package, please amake sure that ROS packages `realsense2_camera` and `darknet_ros` are also running, as mentioned in the repository [Object-Detection-in-ROS](https://github.com/CharmFlex-98/Object-Detection-in-ROS)(Section G)\
After that, Open three terminal:
```
cd ~/catkin_workspace/camera_ws
source devel/setup.bash
roslaunch realsense2_camera rs_camera.launch
```
```
cd ~/catkin_worspace/object_detection_ws
source devel.setup.bash
roslaunch darknet_ros darknet_ros.launch
```
```
cd ~/catkin_worspace/object_detection_ws
source devel/setup.bash
rosrun object_tracking object_tracking.py
```
