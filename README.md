# group_project_7028

Our project main function is to generate the Chinese style painting. So far, we can convert photo (landscape, flower and bird), edge line draft, and other style painting to the Chinese style painting.

## Step 1

```shell
cd ./ui
```

## Step 2
Install requirement package

```shell
pip install requirements.txt
```

You also need to install `tkinter` (cannot use pip to install)
For mac
```shell
$ brew install python-tk
```
After use the brew to install but not work for other package like `pillow`. You can try following to create the venv environment

```shell
python3 -m venv myenv 
# create the venv called myenv
```

```shell
source myenv/bin/activate #activate the venv
```
or

```shell
conda create --name <myenv> # create a new conda env
```

Install other required packages for our project

``` shell
python3 -m pip install requirements.txt 
# try to install all required packages
```
---
For linux
```shell
$ sudo apt-get install python3-tk
```

## Step 3

Our models and dataset: [Google Drive](https://drive.google.com/drive/folders/1cofnlgMdhURPpFCfaryP-wl1sUuUjo35)


Run the code
Putting the model into a checkpoints file with following sturcture
```
checkpoints
├── lastest_net_G
│   └── lastest_net_G.pth
├── lastest_net_G_flowers
│   └── lastest_net_G.pth
└── lastest_net_G_nonchar
│   └── lastest_net_G.pth
└── generator_45_canny.pth
```



If you want to train the model please use
```shell
python3 train.py
```

## Step 4
Run the code to show the final generator result

```shell
python3 tk_test.py
```

![Current UI Demo](/demo.png "Current UI Demo")

Here is some explain for the button in the UI

- `Edge Detect`: Gain the edge line picture for the input image
- `Photo`: Generating the Chinese style painting using 5he original pre-trained model
- `Flower Photo`: Generating flower and bird style painting will be more colorful and bright
- `Nonchar Photo`: Comparing with original generated painting, we removed character in the generated graph
- `Other Type`: It can convert not only photo, but also paintings and edge draft lines


---

### Fine tune
The following steps are used to run the fine tune.

```shell
cd ./ui/finetune
```

Run:

```shell
python3 finetune.py --dataroot ./datasets/dataset1 --pretrain_path /mnt/latest_net_G.pth --checkpoints_dir ./checkpoints 
```

Explanation for above runing code:

`--dataroot` add the dataset path 

`--pretrain_path` add the pretrained model path

`--checkpoints_dir` add the save model checkpoints path

