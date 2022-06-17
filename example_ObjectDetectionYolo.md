# EXAMPLE REPOSITORY
![yolov4](/obj_det.jpeg)
# Object Detection using Yolov3 in ROS
This is a guide on showing how to perform object detection in ROS. Specifically, we want to detect **pedestrian** and **traffic light (red, yellow, green, off, wait on)**. In order to accomplish this, the simplified version of the workflow is as below:

1. Gather datasets from internet
2. Change the datasets to YOLO format 
3. Train datasets and obtain weight file in Darknet 
4. Use the weight file to do object detection in ROS

## Table of Content
Below is the whole procedures that we are going to do in order to do object detection in ROS, from start to end.
The detailed explanation will be summarized in each section respectively.\

A. Download This Repository\
B. Install Virtual Environment for Python 3\
C. Obtain Datasets\
D. Create Train, Valid and Test datasets.\
D. Install Yolov3 ROS Package\
E. Object Dectection code implemented in Python\
F. Run Object Detection node in ROS\

## A. Download This Repository
### Explaination
This repository contains most of the scripts that we need to perform some tasks. Specifically, they are:
* format_handler.py (change datasets to YOLO format)
* Flip_Image.py (flip image datasets to produce more images)
* Rotate_Image.py (rotate image datasets to produce more images)
* visualize.py (to visualize annotated bounding box of the image datasets)
* utils.py (store some extra functions)

We need to download this repository for the utilities that we are going to use in the future.
### Method
Open terminal:
```
cd ~
git clone https://github.com/CharmFlex-98/Object-Detection-in-ROS.git
```
**DEMO VIDEO**\
https://user-images.githubusercontent.com/77537132/138455084-e7a85797-bc66-43ff-b820-7f04a0129b99.mp4
## B. Install Virtual Environment
### Explanation
Since some of the scripts that we are going to use utilize python3, we need to make a virtual environment to store independent libraries and run with python3.
### Method
Open terminal:
```
cd ~
pip install virtualenv
mkdir python_venv
cd python_venv
virtualenv -p /usr/bin/python3.6 python36_venv
echo "alias python36_venv='source ~/python_venv/python36_venv/bin/activate'">>~/.bashrc
source ~/.bashrc
```
Activate the virtual environment: python3_venv\
`python36_venv`

Install some libraries\
`pip install opencv-python`\
`pip install numpy`\

**DEMO VIDEO**\
https://user-images.githubusercontent.com/77537132/138455544-8c69e2e4-2338-4044-aeb4-243acd64f0ab.mp4
## C. Obtain Datasets from Internet
### Explanation
We will use Yolov3 to perform object detection on `Person`, `Car`, and `Traffic light`.\
Both datasets obtained from [OpenImages](https://storage.googleapis.com/openimages/web/visualizer/index.html?set=train&type=detection&c=%2Fm%2F01g317), and their annotations are not in YOLO format. \
In this section, we will 
* create `class.txt` to store name of classes.
* collect both image datasets and change their annotations format into YOLO format, 
* gather them into a folder (full_datasets)
* perform data augmentation to increase the number of datasets to increase accuracy for training.

At the end of this section, we will have `class.txt`, and `full_datasets` folder containing all the images and annotation files for training.

### Method
#### 1. Create `class.txt`
We need to create `class.txt` to store the name of classes (Person, Car, Traffic light).\
`class.txt` will be used in the future.

First, open terminal and activate virtual environment,
```
python36_venv
```
Go to our main directory `Object-Detection-in-ROS`, create `class.txt` and store the names of classes inside.
```
cd ~/Object-Detection-in-ROS
touch class.txt
echo -e 'Person\nCar\nTraffic light'>>class.txt
```
#### 2. Download Dataset
#### 2.1.1 Collect Traffic Lights Datasets (OPTIONAL)
**Only if you want to download seperated traffic light datasets. Else, skip to 2.2.1**
The source of can be found [here](https://github.com/Thinklab-SJTU/S2TLD). For conveniences, the download links are provided as below.\
[S2TLD（1080x1920）](https://1drv.ms/u/s!Akhz5L4oxpUGiX2BR8RRl4B-XJ4I?e=GlYMWJ) (1080 height, 1920 width)

Open the downloaded folder, `S2TLD（1080x1920` and go to the directory where `Annotations`, `JPEGImages` located. `Annotations` folder contains  annotations information of each dataset images, whereas `JPEGImages` contain all the dataset images.\
Copy and paste them into our `Object-Detection-in-ROS` directory. \
Note that:
* The annotations are in xml format. We need to change to yolo format.
* The names of images contain space. This is not a desire thing and we need to rename all of them.
#### 2.1.2 Convert annotation into YOLO format
We will use `format_handler.py` from this repository to change annotations into YOLO format and rename the images.\
First, create a folder `TrafficImages` to store the output YOLO annotations files and the dataset images that will be renamed.
```
mkdir TrafficImages
```
Next, change the annotations format into YOLO and rename the images by running the command below. The output files are all stored in `TrafficImages` folder.
```
python format_handler.py -type traffic -i JPEGImages -a Annotations -o TrafficImages 
```
#### 2.2.1 Collect Person, Car, Traffic light from Open Images
We are going to use the datasets from `Open Images` website. The source can be found [here](https://storage.googleapis.com/openimages/web/visualizer/index.html?set=train&type=detection&c=%2Fm%2F01g317). \
Download datasets from `Open Images` directly might be troublesome and difficult to handle. Thus, we are going to use `OIDv4_ToolKit`. The source is [here](https://github.com/EscVM/OIDv4_ToolKit).
`OIDv4_ToolKit` allows us to download specific class (or combination of classes) of datasets from `Open Images`, and also allow us to choose how many images we want to download.\

Activate the virtual environment: python3_venv\
`python36_venv`

To install `OIDv4_ToolKit`:
```
cd ~/Object-Detection-in-ROS
git clone https://github.com/EscVM/OIDv4_ToolKit.git
cd OIDv4_ToolKit
pip install -r requirements.txt
```
in **OIDv4_ToolKit** folder Download the **Person, Car, Traffic light** datasets from Open Images, and limit our datasets number to 700, as a follows:

```
python main.py downloader --classes Person Car Traffic\ light --type_csv validation --limit 700 --multiclasses 1
```
Preferbly, we need 2000 images for each class. If we have less image we can increase the number of imagen following the (section: How to Improve Object Detection) [ Source](https://github.com/AlexeyAB/darknet).
But since we will use data augmentation method to increase the number of images for training, 1000 images should be fine.

Enter `y` for both prompted message.\
![message](/prompted_msg.png)\

____________________
##### ***[Note: In the case, you cannot download the dataset, please follow the next step:]***
If the processs stopped automatically without downloading the `Dataset` folder, re-run again.
```
python main.py downloader --classes Person Car Traffic\ light --type_csv validation --limit 700 --multiclasses 1
```
If still not working, your device might not capable to download the datasets. Please run the OIDv4_ToolKit in google colab environment \
Go to [google colab](PedestrainImages/train/Person/Label), upload OIDv4_Toolkit_in_Colab.ipynb from this repository, and run all the cells).\
`Dataset.zip` will be downloaded. After completed downloading, unzip the folder and put in `/home/user/Object-Detection-in-ROS` directory.\

____________________

##### ***[Note: In the case, you download the dataset, please follow the next step:]***
The `Dataset` folder containing the **Person, Car, Traffic light** datasets will be created and locate in 
```
/home/user/Object-Detection-in-ROS/OIDv4_ToolKit/OID/
```
The images are located in 
```
/home/user/Object-Detection-in-ROS/OIDv4_ToolKit/OID/Dataset/train/Person_Car_Traffic light
```
the annotation files are store in 
```
/home/user/Object-Detection-in-ROS/OIDv4_ToolKit/OID/Dataset/train/Person_Car_Traffic light/Label
```
Note: we are going to have image and annotation with the same name in .jpg and txt respectively 

Using mv command, move the `Dataset` folder to `Object-Detection-in-ROS` directory.
```
cd ~/Object-Detection-in-ROS
mv OIDv4_ToolKit/OID/Dataset ./
```
____________________

#### 2.2.2 Convert annotation into YOLO format

Next, we will change the datasets in `Dataset` into YOLO format, using the `format_handler.py` from this repository.\
```
python format_handler.py -type OpenImages -i Dataset/train/Person_Car_Traffic\ light -o Dataset/train/Person_Car_Traffic\ light -a Dataset/train/Person_Car_Traffic\ light/Label
```
Now, the **Label** folder in `Dataset/train/Person_Car_Traffic\ light/Label` is no longer needed. Delete it.
```
rm -rf Dataset/train/Person_Car_Traffic\ light/Label
```
Create `full_datasets` folder to store all the datasets.
```
mkdir full_datasets
mv Dataset/train/Person_Car_Traffic\ light/* full_datasets/
```
To view the bounding boxes:
```
python visualize.py -i full_datasets -c class.txt
```
#### 3 Collect Extra Datasets (ONLY datasets from Open Images)
If you need to collect extra datasets of different classes other than provided here, please follow this.\
3.1) Modify class.txt in section [C] with the new  <**extra**> classes as a follows:
```
echo -e 'Person\nCar\nTraffic light\n<Extra1>\n<Extra2>'>class.txt
```
3.2) For Extra class, download new dataset from Open image; (limit 700) means 700 images as following in section [2.2.1] **Person** example.
go to `OIDv4_ToolKit` directory:
```
cd ~/Object-Detection-in-ROS/OIDv4_ToolKit
```
```
python main.py downloader --classes <Extra> --type_csv train --limit 700
```
Move the folder into our main directory and rename the folder .
```
cd ~/Object-Detection-in-ROS
mv OIDv4_ToolKit/OID/Dataset <Extra>Images
```
And repeat as in [2.2.2]. 
```
python format_handler.py -type <Extra> -i <Extra>Images/train/<Extra> -o <Extra>Images/train/<Extra> -a <Extra>Images/train/<Extra>/Label
rm -rf <Extra>Images/train/<Extra>/Label
```

***

#### 4 Combine the extra datasets with the original datasets into folder (full_datasets). (OPTIONAL)
**Only if you have extra datasets**
```
mv <Extra>Images/train/<Extra>/* ./full_datasets/
```
**Congrats**\
The `full_datasets` folder contains all the images and annotation files 

#### 5 Filtering Out Unsuitable Images
To increase the quality of datasets for training, we need to make sure that every target target object in every image had been labelled, and had been labelled properly!\
We can check each image. If ok, press `SPACE`. Else, to delete, press `p`
```
python visualize.py -i full_datasets -c class.txt
```

#### 6 Data Augmentation
Make sure only proceed this after filtering is done\
To increase accuracy of object detection, we need to increase the images for training. For the images in `full_datasets` folder,\
Create a flipped version (total images will be x2 after flipping):
```
python Flip_Image.py full_datasets full_datasets 
```
Create a rotated version (-45 degree to 45 degree), (total images will be x2 after flipping):
```
python Rotate_Image.py full_datasets full_datasets -min -45 -max 45

```
To check the bounding boxes:
```
python visualize.py -i full_datasets -c class.txt
```

**DEMO VIDEO**\

## D. Create Train, Valid and Test Datasets in the Server
### Explanation
Currently, all the images and annotation datasets are stored in `full_datasets`  folder. However, not all of them will be trained, some of them will be used for validation during training process, and some of them will be used for results testing.\
In this section, we will create `txt` file such as:\
`train.txt` contains path to image datasets for training.\
`valid.txt` contains path to image datasets for validation.\
`test.txt` contains path to image datasets for testing.\
At the end of this section, we will make a folder, `my_data` to store `class.txt`, `train.txt`, `valid.txt`, and`test.txt`
### Method
Open a new tab in tmux and SSH into our server:
```
tmux
ssh nabih@10.201.159.235 -p24205
```
Move the whole `Object-Detection-in-ROS` folder from our computer into server computer.
At our own computer terminal:
```
scp -P24205 -r ~/Object-Detection-in-ROS nabih@10.201.159.235:~/
```
At the server, create a virtual environement for python3, as in [B]
```
cd ~
pip install --user virtualenv
mkdir python_venv
cd python_venv
virtualenv -p /usr/bin/python3.6 python36_venv
echo "alias python36_venv='source ~/python_venv/python36_venv/bin/activate'">>~/.bashrc
source ~/.bashrc
```
Activate the virtual environment and install opencv:
```
python36_venv
pip install opencv-python
```
We will create train.txt, valid.txt and test.txt.
In the server:
```
cd ~/Object-Detection-in-ROS
touch train.txt valid.txt test.txt
```
The distribution of datasets for train, valid and test set is 7-2-1 (70% for training, 20% for validation, 10% for testing).\
The images are randomly splitted into each category and their paths are listed inside respectively.
```
python utils.py -type split -d 0.7 0.2 0.1 -i full_datasets
```
Finally, create `my_data` folder to store all the necessary files for training.
```
mkdir my_data
cp class.txt ./my_data/obj.names
cp train.txt ./my_data/train.txt
cp valid.txt ./my_data/valid.txt
cp test.txt ./my_data/test.txt
echo -e 'classes = 3\ntrain = my_data/train.txt\nvalid = my_data/valid.txt\nnames = my_data/obj.names\nbackup = backup/'>>./my_data/obj.data
```

**ONLY IF YOU HAVE EXTRA CLASSES**\
If you have extra classes, edit obj.data in the `my_data` folder,
```
nano my_data/obj.data
```
change line: **classes = number of classes**\
And save:
```
ctrl x -> y -> enter
```


**DEMO VIDEO**\
https://www.youtube.com/watch?v=HiDUvdZNCjo
## E. Install Darknet and Training
There are some dependencies we need to install first before able to install darknet.
### Install OpenCV
```
pip install python-opencv
```
### Install CUDA
First, check the compute capability of your GPU [here](https://en.wikipedia.org/wiki/CUDA). \
Then download the corresponding version of CUDA [here](https://developer.nvidia.com/cuda-toolkit-archive). \
For the sake of convenience, choose deb (local) as the installation type. The installation guide is provided once you have selected the installation type.
### Install cuDNN
Download the cuDNN [here](https://developer.nvidia.com/rdp/cudnn-archive)
. \
Note that you need to select the appropriate version corresponding to your CUDA version.\
After downloading completed, navigate to the directory containing cuDNN tar file.\
Open a new terminal in your computer, and copy the file to server:
```
scp -P24205 -r ./cudnn-11.0-linux-x64-v8.0.5.39.tgz nabih@10.201.159.235:~/
```
In the server, unzip the tar file

```
tar -xzvf cudnn-x.x-linux-x64-v8.x.x.x.tgz
```
and copy some files into CUDA directory:
```
sudo cp cuda/include/cudnn*.h /usr/local/cuda-11.0/include
sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda-11.0/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda-11.0/lib64/libcudnn*
```
*Note that the `x` follows the name of tar file.
### Install Darknet
We use [darknet](https://github.com/AlexeyAB/darknet) from AlexeyAB's version.
```
cd ~
git clone https://github.com/AlexeyAB/darknet.git
```
In the darknet directory, open the `Makefile`:
```
cd darknet
nano Makefile
```
and set:
```
GPU=1
CUDNN=1
CUDNN_HALF=1
OPENCV=1
LIBSO=1
```
Save `Makefile`:
```
ctrl x -> y -> enter
```
Export the cuda paths that we use (in our case, it is cuda 11.0):
```
export PATH=/usr/local/cuda-11.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```
Finally, compile the darknet:
```
cd darknet
make
```
**DEMO VIDEO**\
https://www.youtube.com/watch?v=olTteetQBgU
### Train Datasets
Now that we have darknet compiled and installed, we can start training now.\
First, download pretrained weights [darknet53.conv.74](https://pjreddie.com/media/files/darknet53.conv.74) into our computer.\
In directory where **darknet53.conv.74** stored, open terminak and copy the file to the server.
```
scp -P24205 -r ./darknet53.conv.74 nabih@10.201.159.235:~/
```
In the server terminal, create `weights` folder and move the **darknet53.conv.74** into the folder.
```
cd ~/darknet
mkdir weights
mv ~/darknet53.conv.74 weights/
```
Now, move on to our own datasets training.\
Copy `my_data` into darknet directory.
```
cp -r ~/Object-Detection-in-ROS/my_data ./
```
Make two copies of yolov3.cfg in `darknet/cfg`. This `yolov3.cfg` is originally located in the `darknet/cfg` folder when darknet is compiled.、
Rename the two copies of `yolov3.cfg`, one for training (**yolov3_train.cfg**), and one for testing (**yolov3_test.cfg**).\
```
cd cfg
cp yolov3.cfg yolov3_train.cfg
cp yolov3.cfg yolov3_test.cfg
```
We need to change the parameters in `yolov3_train.cfg` and `yolov3_test.cfg` according to the number of classes we want to train.\
Below are the parameters we need to change and its explanation:
```
batches
subdivision
max_batches  (means the iterations for training)
steps
width, height (these two values must be identicle. The larger the value, the higher the accuracy for training.)
classes  (There are 3 `classes` that we need to change, under each [yolo] layer)
filters  (There are 3 `filters` that we need to change, under [convolutional] layer exact before each [yolo] layer)
```
Open **yolov3_train.cfg**:
```
nano yolov3_train.cfg
```
Edit **yolov3_train.cfg**:
```
batch=64
subdivisions=16
max_batches=12000         # classes*2000, but not less than number of training images, and 6000
steps=9600, 10800         # 80% and 90% of max_batches
width=416, height=416     # or any mutiple value of 32
classes=6                 # number of classes, in each of 3 [yolo] layers
filters=33                # (class+5)*3, in last [convolutional] layer before each of 3 [yolo] layers
```
Save **yolov3_train.cfg**:
```
ctrl x > y > enter
```
Open and edit **yolov3_test.cfg**:
```
nano yolov3_test.cfg
```

Edit **yolov3_test.cfg**. Everything is the same as **yolov3_train.cfg** , except:
```
batch=1
subdivisions=1
```
Save **yolov3_test.cfg**:
```
ctrl x -> y -> enter
```
Lastly, since we want intermediate weight file every 1000 iterations so that we can choose best weight among them,\
open detector.c in darknet/src/:
```
nano ~/darknet/src/detector.c
```
and remove the line **&& net.max_batches < 10000** in
```
(iteration >= (iter_save + 1000) || iteration % 1000 == 0) && net.max_batches < 10000)
```
Save detector.c:
```
ctrl x -> y -> enter
```
and recompile darknet
```
cd ~/darknet
make
```
Finally, we can start training!
```
cd ~/darknet
./darknet detector train my_data/obj.data cfg/yolov3_train.cfg weights/darknet53.conv.74 -map
```
**After training completed, you will find severals weigh files such as yolov3_train_X000.weights, yolov3_train_best.weights in the `backup` folder. The weights file which has highest accuracy is labelled as yolov3_train_best.weights. We will use this one for objection detection**\

## F. Install Yolov3 ROS Package on Jetson Nano
After completed training, we obtain the weight file for object detection. To run object detection and communicate with other ROS nodes, we need ROS, OpenCV, CUDA installed in order to successfully run. If you have't installed ROS yet, please follow the [documentation](http://wiki.ros.org/melodic/Installation/Ubuntu).
### Install darknet_ros
There is already ROS package for Yolov3 out there. This is the [source](https://github.com/leggedrobotics/darknet_ros).\
[source2](https://taemian.tistory.com/entry/ROS-1-n-darknetros%EB%A5%BC-%ED%99%9C%EC%9A%A9%ED%95%9C-Yolo-v3-%EC%82%AC%EC%9A%A9%EB%B2%95)
Create a catkin workspace for ROS if haven't done so.
```
mkdir -p ~/catkin_workspace/object_detection_ws/src
cd ~/catkin_workspace/object_detection_ws
catkin_make
```
Go into the source file of catkin workspace and install the darknet_ros package.
```
cd ~/catkin_workspace/object_detection_ws/src
git clone --recursive https://github.com/leggedrobotics/darknet_ros.git
cd ../
catkin_make
```
If error occurs such as 
```
nvcc fatal : Unsupported gpu architecture 'compute_61'.
```
This means that you need to check the compute capability (version) of your GPU. You can find a list of supported GPUs in CUDA here: [CUDA - WIKIPEDIA](https://en.wikipedia.org/wiki/CUDA#Supported_GPUs). Simply find the compute capability of your GPU and add it into `darknet_ros/CMakeLists.txt`. Simply add a similar line like
```
-O3 -gencode arch=compute_50,code=sm_50
```
and remove the one (for example:-gencode arch=compute_61,code=sm_61) which caused error. The installation should have no problem.\
Compile again:
```
cd ~/catkin_workspace/object_detection_ws/
catkin_make
```

**DEMO VIDEO**\

## G. Run darknet_ros node on Jetson Nano
Upon sucessful installation of `darknet_ros` package, we can try running detection by using the weight obtained from previous training. We need to transfer the `yolov3_test.cfg` and `yolov3_train_best.weights` from server to Jetson Nano. (In the demo viideo at the bottom of this section, I demonstrate using yolov3-tiny instead of yolov3. The procedures are still the same)\
Open a terminal in Jetson Nano:
```
scp -P24205 nabih@10.201.159.235:~/darknet/backup/yolov3_train_best.weights ~/catkin_workspace/object_detection_ws/src/darknet_ros/darknet_ros/yolo_network_config/weights
scp -P24205 nabih@10.201.159.235:~/darknet/cfg/yolov3_test.cfg ~/catkin_workspace/object_detection_ws/src/darknet_ros/darknet_ros/yolo_network_config/cfg
```
Then, direct to `darknet_ros/config` folder to edit `yolov3.yaml` to tell the detector where to find our weight file and configuration file for yolov3.\
Go to the `config` folder:
```
cd ~/catkin_workspace/object_detection_ws/src/darknet_ros/darknet_ros/config
```
Open yolov3.yaml, 
```
gedit yolov3.yaml
```
Edit the content so they are the same as below:
```
yolo_model:

  config_file:
    name: yolov3_test.cfg
  weight_file:
    name: yolov3_train_best.weights
  threshold:
    value: 0.3
  detection_classes:
    names:
      - Person
      - Car
      - Traffic light
```
Open **ros.yaml**
```
gedit ros.yaml
```
change the topic of camera reading in `ros.yaml` to **/camera/color/image_raw**, which is the topic of our camera\
Finally, open `darknet_ros.launch` in `launch` folder:
```
cd ~/catkin_workspace/object_detection_ws/src/darknet_ros/darknet_ros/launch
gedit darknet_ros.launch
```
Under **ROS and network parameter files**, change the target yaml file from `yolov2-tiny.yaml` to `yolov3.yaml`
```
  <arg name="ros_param_file"             default="$(find darknet_ros)/config/ros.yaml"/>
  <arg name="network_param_file"         default="$(find darknet_ros)/config/yolov3.yaml"/>
```

Now, we can start to run the detector. Start the Roscore:
```
roscore
```
and open a new tab:
```
source ~/catkin_workspace/object_detection_ws/devel/setup.bash
roslaunch darknet_ros darknet_ros.launch
```
You will notice the node is trying to receive image stream 
![testing](/darknet_ros_testing.png)

Open a new tab to run camera node
```
cd ~/camera_ws
source devel/setup.bash
roslaunch realsense2_camera rs_camera.launch
```
Congratulation. Now you can see the bounding boxes of the target classes from the realsense camera.

**DEMO VIDEO**\
https://youtu.be/w5aVjt9w1YY

## F. Implementing Object Tracking from the Information of Bounding Boxes
We need to refer to other repository. \
Please refer [here](https://github.com/CharmFlex-98/ROS-Object-Tracking)

