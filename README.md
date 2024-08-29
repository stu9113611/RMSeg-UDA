# RMSeg-UDA: Unsupervised Domain Adpatation for Road Marking Segmentation under Adverse Conditions
Reference implementation about RMSeg-UDA

![image](https://github.com/stu9113611/RMSeg-UDA/blob/main/architecture.png)

This repository contains the reference implementation for RMSeg-UDA, an UDA road marking segmentation training framework.

## Table of Contents
1. [Introduction](#introduction)
2. [Environment](#environment)
3. [Dataset](#dataset)
4. [Execution](#execution)

## Introduction

This repository provides a whole framework for training, inference, and evaluation of the road marking segmentation model (SegFormer).
Additionally, a road marking segmentation dataset under clear, night, and rainy conditions was built to support this work.

## Environment

The framework has been tested on the environment:

**System Hardware**:

- CPU: Intel® Core™ i7-14700KF
- Memory: 64.0 GiB
- GPU: Nvidia GeForce RTX 4090

**System Software**:

- OS: Ubuntu 22.04.3 LTS
- CUDA：12.3
- CuDNN：8.7.0.0
- Python：3.11

**Installation**:

First install the newest pytorch distribution. In my case, I use pytorch-2.4.0 and torchvision-0.19.0.
Then install the requirements with
```
pip install -r requirements.txt
```

## Dataset

In this work, a road marking segmentation dataset under clear, night, and rainy conditions was proposed.

**Road Line and Marking Segmentation Dataset under Adverse Contidions (RLMD-AC)** is a extensive version of [RLMD](https://github.com/stu9113611/RLMD). The annotation principles and format are identical.
The images and labels are saved in 1080p, and be seperated into folders with different weather conditions.
Since RLMD-AC was proposed to support RMSeg-UDA, the training sets of night and rainy conditions are **unlabeled**.

Please download RLMD-AC from this [google drive](https://drive.google.com/drive/folders/1ZVxIanHEPq5gPjDbKlmZV3K99SsM1NT-?usp=sharing). After that, please unzip the images and labels.

The dataset is expected to have this folder structure:
```
RMSeg-UDA/
    data/
        rlmd_ac/
            clear/
                train/
                    images/
                    labels/
                val/
                    images/
                    labels/
            night/
                train/
                    images/
                val/
                    images/
                    labels/
            rainy/
                train/
                    images/
                val/
                    images/
                    labels/
```
This rule applies to other datasets, too.

## Execution

Configuration files in JSON foramt are used in this framework. For practical examples, please check the [configs folder](https://github.com/stu9113611/RMSeg-UDA/tree/main/configs).

---
Before training, class statistics file for the clear training data should be generated.
```
python -m tools.count_categories <path/to/your/csv> <path/to/your/training_data_root> <path/to/your/rcs_savepath>
```
For example,
```
python -m tools.count_categories data/csv/rlmd.csv data/rlmd_ac/clear/train data/rlmd_ac/clear/rcs.json
```

---
To train the model, please choose one configuration (or make one yourself), give an experiment name as follow:
```
python -m tools.train <path/to/your/config> <experiment_name>
```
For example, 
```
python -m tools.train configs/train_rlmd_clear_to_rainy.json demo_experiment
```
---
To resume training from an interupted experiment:
```
python -m tools.train <path/to/your/config> <experiment_name> <checkpoint filename>
```
For example, 
```
python -m tools.train configs/train_rlmd_clear_to_rainy.json demo_experiment checkpoint_latest.pth
```
or from a specific timing,
```
python -m tools.train configs/train_rlmd_clear_to_rainy.json demo_experiment checkpoint_20000.pth
```
---
To test the trained model, please choose one configuration, the log directory, and the checkpoint filename:
```
python -m tools.test <path/to/your/config> <path/to/your/log> <checkpoint filename>
```
For example,
```
python -m tools.test configs/train_rlmd_clear_to_rainy.json logs/rlmd/clear_to_rainy/demo_experiment checkpoint_latest.pth
```
---
If you are using your custom dataset, please make sure the labels are stored in P-mode (palette mode), or the code won't work.

Additionally, you should prepare your own category csv file, which should follow the format in the [given ones](https://github.com/stu9113611/RMSeg-UDA/tree/main/data/csv).
```
python -m tools.convert_to_p_mode <path/to/your/category/csv> <path/to/your/labels> <path/to/your/output>
```
---
To inference data, we provide inferencing function for folder and video.

To inference a folder of images:
```
python -m tools.inference_folder <path/to/your/category/csv> <path/to/your/images>\
 <suffix> <path/to/your/output> <path/to/your/checkpoint> <height> <width> <use _sliding_inference>
```
For example,
```
python -m tools.inference_folder data/csv/rlmd.csv data/rlmd_ac/clear/val/images\
 .jpg inference_output logs/rlmd/clear_to_rainy/demo_experiment/checkpoint_latest.pth 1080 1920 --sliding-window
```

To inference a video:
```
python -m tools.inference_video <path/to/your/category/csv> <path/to/your/video>\
 <path/to/your/output> <path/to/your/checkpoint> <height> <width> <output_framerate> <use _sliding_inference>
```
For example,
```
python -m tools.inference_video data/csv/rlmd.csv a_video_footage.mp4\
inference_output.mp4 logs/rlmd/clear_to_rainy/demo_experiment/checkpoint_latest.pth 1080 1920 30 --sliding-window
```
We only support to save mp4 output for now.

---
The framework also provides t-SNE visualization.

First, generate and save the features to be visualized.
```
python -m tools.save_features_for_tsne
```

Second, visualize the features with t-SNE visualization.
```
python -m tools.tsne_visualization
```

The codes in [save_features_for_tsne.py](https://github.com/stu9113611/RMSeg-UDA/blob/main/tools/save_feature_for_tsne.py) and [tsne_visualization](https://github.com/stu9113611/RMSeg-UDA/blob/main/tools/tsne_visualization.py) do not provide argparse feature, please check the exact code, it should be easy to read & be modified.
