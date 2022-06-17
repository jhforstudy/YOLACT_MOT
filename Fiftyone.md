# Tutorials for Fiftyone
As the yolact uses the COCO annotation format and from the [official page](https://cocodataset.org/#download), they provide the large size of dataset with annotation. Annotation for instance segmentation is pretty hard task because it should be drawn several polygons in hand on segmentation. So, by crawling the specific category of the object in COCO, we can obtain the specific dataset in our purpose easily. This method is given by `fiftyone`. From installation of fiftyone and python script of dataset information, fiftyone search the object included image and make annotation of those. This is the source I referenced : [Source](https://medium.com/voxel51/the-coco-dataset-best-practices-for-downloading-visualization-and-evaluation-68a3d7e97fb7)
# Table of Contents
1. Setup
2. Downloading dataset
3. Arguments Description

## 1. Setup 
To setup the fiftyone, you have to activate the virtual enironment of python. You already followed the `Yolact_Tutorial` in previous steps, you have the virtual environment in your laptop. Please activate the virtual environment.\
Activate your virtual environment :
```Shell
conda activate yolact
```
Now install the fiftyone module:
```Shell
pip install fiftyone
```
## 2. Downloading dataset
After install fiftyone, you can launch the fiftyone using web server and this is done by simple python code. The python code contains some blocks of code and as the arguments indicate the dataset style (in this case, dataset split : "train" and specific for catefory : "person", "car", ...) and then search the images and make annotation. So, make the `fiftyone_coco.py` in `~/yolact/data`. The example content is below.
```python
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="train",
    label_types=["detection","segmentations"],
    classes=["person", "car", "truck", "traffic light"],
    max_samples=3000,
)

session = fo.launch_app(dataset)
session.wait()
```
This content means : In `coco-2017` dataset, we will download the `detection` and `segmentation` dataset for `train`,
 and the dataset includes `person`, `car`, `truck`, `traffic light`, while size of sample is 3000.\
Now, launch your python code (**Note : In your virtual environment**) :
```Shell
cd ~/yolact/data
python fiftyone.py
```
This is the visualization of searching images.(This is not the result of above code.)\
![fiftyone](https://user-images.githubusercontent.com/78340346/161711340-0607b583-cacc-430f-beed-6e86e55476d6.png) 

## 3. Arguments Description
From the python script, the web server of the fiftyone search the objective images. So, arguments of the python script decide the dataset contents. You can write your own code by using this arguments information.
Here is arguments description :

* `label_types` : a list of types of labels to load. Values are ("detections", "segmentations"). By default, all labels are loaded but not every sample will include each label type. If max_samples and label_types are both specified, then every sample will include the specified label types.
* `split` and `splits` : either a string or list of strings dictating the splits to load. Available splits are ("test", "train", "validation").
* `classes` : a list of strings specifying required classes to load. Only samples containing at least one instance of a specified class will be downloaded.
* `max_samples` : a maximum number of samples to import. By default, all samples are imported.
* `shuffle` : boolean dictating whether to randomly shuffle the order in which the samples are imported.
* `seed` : a random seed to use when shuffling.
* `image_ids` : a list of specific image IDs to load or a filepath to a file containing that list. The IDs can be specified either as `<split>/<image-id>` or `<image-id>`
    
