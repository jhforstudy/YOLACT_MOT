# Yolact Tutorial
Yolact is a Instance segmentation detecting model. Also, it is a realtime streaming detector by using multi threading process of frames. This repository is a detailed guideline how to install and use the yolact. There are 4 main purposes of tutorials and we will use yolact specifically for detect person.
- Yolact installation (section 1)
- Dataset (section 2,3)
- 3 main procedures of deep-learning detector : Train, Validation, Detection(section 4,5,6)
- YOLACT using RISE-LAB server (section 7)

## Table of Contents
Below is the whole procedures that we are going to do in order to detect the segmentation of objects, from start to end. The detailed explanation will be summarized in each section respectively.
1. Yolact installation
2. COCO Datasets
3. Custom Dataset
4. Train
5. Validation
6. Detection
7. YOLACT using RISE-LAB server

## Prerequisites
* Ubuntu 18.04 
* NVIDIA GPU driver & CUDA 10 or later \
**Note : While installing NVIDIA, the Ubuntu default gpu driver 'nouveau' may conflic with nvidia. If that is your case, add nouveau.nomodeset=0 in grub menu.** : [Source](https://blog.neonkid.xyz/66)
* Anaconda Environment 
* PyTorch

## 1. Yolact installation
### 1.1. Create a python3 based environment
Yolact operates in python3 based virtual environment, so create the virtual environment using anaconda.In the case of using server in section 7, we will use virtualenv module instead because of some authority problems in server. Referenced in source [2]
```Shell
conda create -n yolact python=3.6
```

### 1.2. Install Pytorch 1.0.1 or higher and TorchVision
Yolact use pytorch and torchvision which contains basic and useful deeplearning methods. Please check your pytorch version from [here](https://pytorch.org/get-started/locally/), and **find your installation command and run on your terminal**(`Run this command`) : 
![Run this command](https://user-images.githubusercontent.com/78340346/157592768-b90429c1-2c6c-4e25-a5f5-4c1e41693d78.png) \
Then type in the terminal :

```Shell
pip3 install torch torchvision
```

### 1.3. Install other packages and modules
Here are some other installations of package for using yolact. (**Note** : Cython needs to be installed before pycocotools!)
```Shell
pip install cython
pip install opencv-python pillow pycocotools matplotlib
```

### 1.4. Clone git-hub resources
Finally clone the yolact resources from github repository.
```Shell
git clone https://github.com/dbolya/yolact.git
```

## 2. COCO Datasets
Yolact training model uses the `COCO` annotation format. Yolact github page offers the COCO dataset which include 80 categories(person, bear,...) of objects and its segmentic annotation[3]. This COCO dataset is widely used in train so let me introduce how to download it. There are 2 ways of download COCO dataset, so, choose and follow one of them. (**Note** : If you want to use pre-trained weight file, you may skip this topic. Because it takes long time for download the dataset as there are large image files in dataset.)
### 2.1. Download using sh scripts
From the sh file in yolact sources, you can download the COCO dataset. While download the dataset, your internet connection should be stable. They include  2014/2017 year training datasets.
```Shell
sh data/scripts/COCO.sh
```
Here is the command of download COCO test dataset.
```Shell
sh data/scripts/COCO_test.sh
```

### 2.2. Download using links
You can download the COCO dataset from coco official website : [cocodataset.org](http://cocodataset.org/#download) \

For individual download of dataset:\
- [2017 Train images [118K/18GB]](http://images.cocodataset.org/zips/train2017.zip)

- [2017 Val images [5K/1GB]](http://images.cocodataset.org/zips/val2017.zip)

- [2017 Test images [41K/6GB]](http://images.cocodataset.org/zips/test2017.zip)

- [2017 Train/Val annotations [241MB]](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)

- [2014 Train/Val annotations [241MB]](http://images.cocodataset.org/annotations/annotations_trainval2014.zip)

- [2017 Testing Image info [1MB]](http://images.cocodataset.org/annotations/image_info_test2017.zip)

**Note** 
- After Download and move the raw image files directly to the `~/your_path/yolact/data/coco/images`\
- After Download and move the annotation files directly to the `~/yourpath/yolact/data/coco/annotations`

## 3. Custom dataset
### 3.1. Search the specific catogories of dataset from COCO.
This is the introduction of collecting dataset using `fiftyone`. You can simply crawl the specific categories of images in COCO dataset and obtain annotations of searched images by `fiftyone`. As the semantic annotation takes much effort and time(making polygons of every objects), this is very comfortable way. Here is the [fiftyone tutorial](https://github.com/nabihandres/YOLACT/blob/main/fiftyone.md)

**Note** : After Download and move the raw image files directly to the ~/your_path/yolact/data/coco/images\
**Note** : After Download and move the annotation files directly to the ./yourpath/yolact/data/coco/annotations\

Also, you have to make configuration of custom dataset for training. Here is the dataset section, so configurations are dealed in Train section.
### 3.2. Annotating tools to make segmentation by hand.
You can make the custom dataset from raw images by denote the segmentation by hand. This is the case of making your own dataset in a raw images and I recommend the `labelme` as the annotation tool. Here is the [source](https://github.com/wkentaro/labelme).

## 4. Train
### 4.1. Download backbone
Now your dataset is ready in decent directory in yolact. You can train the `dataset` with `backbone` and finally get the weight file(usually `.pth`). The weight file remember the feature of the object and used in detector section. Here are backbones to be used in train, so please download one of them to the directory : `~/yolact/weights`. If you doesn't have the `weights` directory please make that.\
Backbones :\
- Resnet101, download `resnet101_reducedfc.pth` from [here](https://drive.google.com/file/d/1tvqFPd4bJtakOlmn-uIA492g2qurRChj/view?usp=sharing).
- Resnet50, download `resnet50-19c8e357.pth` from [here](https://drive.google.com/file/d/1Jy3yCdbatgXa5YYIdTCRrSV0S9V5g1rn/view?usp=sharing).
- Darknet53, download `darknet53.pth` from [here](https://drive.google.com/file/d/17Y431j4sagFpSReuPNoFcj9h7azDTZFf/view?usp=sharing).

### 4.2. Train dataset with backbone.
From your dataset and backbone, you can train the weight file. Also each train needs parameters of yolact system, these are defined in `config.py`. Here is simple running commands to train.
**Note** 
- You can press ctrl+c while training and it will save an `*_interrupt.pth` file at the current iteration.
- All weights are saved in the `./weights` directory by default with the file name `<config>_<epoch>_<iter>.pth`.

```Shell
# Trains using the base config with a batch size of 8 (the default).
python train.py --config=yolact_base_config

# Trains yolact_base_config with a batch_size of 5. For the 550px models, 1 batch takes up around 1.5 gigs of VRAM, so specify accordingly.
python train.py --config=yolact_base_config --batch_size=5

# Resume training yolact_base with a specific weight file and start from the iteration specified in the weight file's name.
python train.py --config=yolact_base_config --resume=weights/yolact_base_10_32100.pth --start_iter=-1

# Use the help option to see a description of all available command line arguments
python train.py --help
```

It needs 1.5GB GPU memory for 1 batch_size, the training speed goes better as batch_size bigger.
It may takes long time for train 164,000 COCO image datas.
If you want to check your GPU memory usage :
```Shell
nvidia-smi
```

### 4.3. Train with your custom dataset
After you have your custom dataset, you should add some configure informations on `config.py` before use your dataset. 
the path from master is `./data/config.py`.
you should add below code under `dataset_base` definition in `config.py`

```python
my_custom_dataset = dataset_base.copy({
    'name': 'My Dataset',

    'train_images': 'path_to_training_images',
    'train_info':   'path_to_training_annotation',

    'valid_images': 'path_to_validation_images',
    'valid_info':   'path_to_validation_annotation',

    'has_gt': True,
    'class_names': ('my_class_id_1', 'my_class_id_2', 'my_class_id_3', ...)
    
})
```

- A couple things to note:
     - Class IDs in the annotation file should start at 1 and increase sequentially on the order of class_names. If this isn't the case for your annotation file (like in COCO), see the field label_map in dataset_base.
     - If you do not want to create a validation split, use the same image path and annotations file for validation. By default (see python train.py --help), train.py will output validation mAP for the first 5000 images in the dataset every 2 epochs.
- Finally, in yolact_base_config in the same file, change the value for 'dataset' to 'my_custom_dataset' or whatever you named the config object above. Then you can use any of the training commands in the previous section.


## 5. Validation
### 5.1. Download the pre-trained model
In the lsat section, I introduced the method of dataset and train. From this method, you can obtain your own weight file. Also, here are some open source pretrained weight files in YOLACT. It is provided in yolact github page[1]. Also, pretrained weights are released on April 5th, 2019 along with their FPS on a Titan Xp and mAP on `test-dev`: 

| Image Size | Backbone      | FPS  | mAP  | Weights                                                                                                              |  |
|:----------:|:-------------:|:----:|:----:|----------------------------------------------------------------------------------------------------------------------|--------|
| 550        | Resnet50-FPN  | 42.5 | 28.2 | [yolact_resnet50_54_800000.pth](https://drive.google.com/file/d/1yp7ZbbDwvMiFJEq4ptVKTYTI2VeRDXl0/view?usp=sharing)  | [Mirror](https://ucdavis365-my.sharepoint.com/:u:/g/personal/yongjaelee_ucdavis_edu/EUVpxoSXaqNIlssoLKOEoCcB1m0RpzGq_Khp5n1VX3zcUw) |
| 550        | Darknet53-FPN | 40.0 | 28.7 | [yolact_darknet53_54_800000.pth](https://drive.google.com/file/d/1dukLrTzZQEuhzitGkHaGjphlmRJOjVnP/view?usp=sharing) | [Mirror](https://ucdavis365-my.sharepoint.com/:u:/g/personal/yongjaelee_ucdavis_edu/ERrao26c8llJn25dIyZPhwMBxUp2GdZTKIMUQA3t0djHLw)
| 550        | Resnet101-FPN | 33.5 | 29.8 | [yolact_base_54_800000.pth](https://drive.google.com/file/d/1UYy3dMapbH1BnmtZU4WH1zbYgOzzHHf_/view?usp=sharing)      | [Mirror](https://ucdavis365-my.sharepoint.com/:u:/g/personal/yongjaelee_ucdavis_edu/EYRWxBEoKU9DiblrWx2M89MBGFkVVB_drlRd_v5sdT3Hgg)
| 700        | Resnet101-FPN | 23.6 | 31.2 | [yolact_im700_54_800000.pth](https://drive.google.com/file/d/1lE4Lz5p25teiXV-6HdTiOJSnS7u7GBzg/view?usp=sharing)     | [Mirror](https://ucdavis365-my.sharepoint.com/:u:/g/personal/yongjaelee_ucdavis_edu/Eagg5RSc5hFEhp7sPtvLNyoBjhlf2feog7t8OQzHKKphjw)

**Note** : To evaluate the model, you should download the weight file to directory : `~/yolact/weights`. The config name is everything before the numbers in the file name. For example, config is `yolact_base` for `yolact_base_54_8000000.pth`. (Also, 54 : epoch, 800000 : iteration)

### 5.2. Validation of performance
To check the performance of your weight fiel, you can do the validation steps. Also the training steps contrains process calidation in each epochs, this validation step is measuring steps of final weight file.
```Shell
# Quantitatively evaluate a trained model on the entire validation set. Make sure you have COCO downloaded as above. This should get 29.92 mAP in lastly checked.
python eval.py --trained_model=weights/yolact_base_54_800000.pth

# Output a COCOEval json to submit to the website or to use the run_coco_eval.py script.
# This command will create './results/bbox_detections.json' and './results/mask_detections.json' for detection and instance segmentation respectively.
python eval.py --trained_model=weights/yolact_base_54_800000.pth --output_coco_json

# You can run COCOEval on the files created in the previous command. The performance should match my implementation in eval.py.
python run_coco_eval.py

# To output a coco json file for test-dev, make sure you have test-dev downloaded from above and go
python eval.py --trained_model=weights/yolact_base_54_800000.pth --output_coco_json --dataset=coco2017_testdev_dataset

# Display qualitative results on COCO. From here on I'll use a confidence threshold of 0.15.
python eval.py --trained_model=weights/yolact_base_54_800000.pth --score_threshold=0.15 --top_k=15 --display
```
## 6. Detection
Here is the description of how to launch the detecting files. Yolact provide the 2 main detecting methods : `Image`, `Video`. Detection steps use `eval.py` and the method of detection is determined by input parameter --image and --video etc.
### 6.1. Images
Here is the detecting steps for raw image files. You can detect the object and display the segment results.\ 
Input images can be given in : One raw image, Image directory. Also, you can save the output file py give parameter like this : `--image=input_image:output_image.png`. Here are the command.
```Shell
# Display detections on the specified image.
python eval.py --trained_model=weights/yolact_base_54_800000.pth --score_threshold=0.15 --top_k=15 --image=my_image.png

# Process an image and save it to another file.
python eval.py --trained_model=weights/yolact_base_54_800000.pth --score_threshold=0.15 --top_k=15 --image=input_image.png:output_image.png

# Process a whole folder of images.
python eval.py --trained_model=weights/yolact_base_54_800000.pth --score_threshold=0.15 --top_k=15 --images=path/to/input/folder:path/to/output/folder
```
These are results of the detection in images\
<img src=https://user-images.githubusercontent.com/78340346/157192987-f118bdb0-987e-4f2c-aac4-8e05368e2422.png width=400 height=400>
<img src=https://user-images.githubusercontent.com/78340346/157193000-3681a4e1-f628-4571-9d37-a01e39cac669.png width=400 height=400>

### 6.2. Video
The method of real-tiem detecting and detection on saved video file is given in this section. The parameter is `--video` and by `--video_multiframe`, you can process the video inputs in parallel thread. Basic multithread is 4 and the detecting speed may differ from this argument. Here is video detecting commands.
```Shell
# Display a video in real-time. "--video_multiframe" will process that many frames at once for improved performance.
# If you want, use "--display_fps" to draw the FPS directly on the frame.
python eval.py --trained_model=weights/yolact_base_54_800000.pth --score_threshold=0.15 --top_k=15 --video_multiframe=4 --video=my_video.mp4

# Display a webcam feed in real-time. If you have multiple webcams pass the index of the webcam you want instead of 0.
python eval.py --trained_model=weights/yolact_base_54_800000.pth --score_threshold=0.15 --top_k=15 --video_multiframe=4 --video=0

# Process a video and save it to another file. This uses the same pipeline as the ones above now, so it's fast!
python eval.py --trained_model=weights/yolact_base_54_800000.pth --score_threshold=0.15 --top_k=15 --video_multiframe=4 --video=input_video.mp4:output_video.mp4
```


## 7. YOLACT using RISE-LAB server
You can use the `RISE-LAB server` for YOLACT works. Here are the basic usages of using the server. 
### 7.1 Install ssh and tmux
**Install ssh**
[Source](https://www.cyberciti.biz/faq/ubuntu-linux-install-openssh-server/).
```
sudo apt install openssh-server
```
```
sudo systemctl enable ssh.service
```
```
sudo systemctl start ssh.service
```
```
sudo dpkg-reconfigure openssh-server
```
**Install tmux**
[Source](https://linuxize.com/post/getting-started-with-tmux/).
```
sudo apt-get install tmux
```
### 7.2. Create virtual environment and install packages on server
Open a new tab in tmux and SSH into our server :
```Shell
tmux
ssh nabih@10.201.159.235 -p24205
```
At the server, create a virtual environment for python3.(If you already have the same named virtual env in server directory, use different env name)
```Shell
cd ~
pip install --user virtualenv
mkdir python_yolact_venv
cd python_yolact_venv
virtualenv -p /usr/bin/python3.6 python36_yolact_venv
echo "alias python36_yolact_venv='source ~/python_yolact_venv/python36_yolact_venv/bin/activate'">>~/.bashrc
source ~/.bashrc
```
Activate the virtual environment and install prerequisite packages:
```Shell
python36_venv
pip3 install torch torchvision
pip install cython
pip install opencv-python pillow pycocotools matplotlib
```
### 7.3. Move your yolact directory to the server
Move the whole `yolact` folder from our computer into server computer.
At our own computer terminal:
```Shell
scp -P24205 -r ~/yolact nabih@10.201.159.235:~/
```
Now you can do the same process to use YOLACT introduced in previous topics. But in my case, I only used for train to use GPU of the server.

### 7.4. ShortCut After install prerequisites.
```Shell
# Train using Server
tmux
ssh nabih@10.201.159.235 -p24205
python36_yolact_venv
cd ~/yolact
python train.py --config=yolact_custom_config         # default batch_size = 8
```
You can train by larger batch_size for train and this may make the training much faster. But too large batch_size mean lower detect performance, so recommand default batch_size=8.
For resuming the train, for example :
```Shell
python train.py --config=yolact_custom_config --resume=weights/yolact_custom_10_32100.pth --start_iter=-1
```

### 7.5. Basic commands for using server
```Shell
# Connect Rise Lab Server
tmux
ssh nabih@10.201.159.235 -p24205

# Copy files : PC -> server
scp -P24205 ~/<file.py> nabih@10.201.159.235:~/<path server destination>

# Copy Directory : PC -> server
scp -P24205 -r ~/<directory> nabih@10.201.159.235:~/<path server destination>

# Copy files : server -> PC
scp -P24205 nabih@10.201.159.235:~/<file.py> ~/<path pc destination>

# Copy Directory : server -> PC
scp -P24205 -r nabih@10.201.159.235:~/<directory> ~/<pc_directory>

```

### 7.6. Error issues
* While doing the training, you may have the error message as cuda and cpu devices can't operated together. Then see this [page](https://github.com/dbolya/yolact/issues/664)\
In my case, i added the `generator` attribute at data_loader
```python
data_loader = data.DataLoader(... generator=torch.Generator(device='cuda'))
```


## References
[1] dbolya/yolact: A simple, fully convolutional model for real-time instance segmentation. https://github.com/dbolya/yolacthttps://github.com/dbolya/yolact, GitHub, June 9, 2022\
[2] yolact instance segmentation algorithm source code running tutorial, https://blog.katastros.com/a?ID=01200-2bc8e489-6a21-4d9f-98c3-295c7cf593eb, Katastros, June 9, 2022\
[3] Common Objects in Context, https://cocodataset.org, COCO, June 9, 2022


