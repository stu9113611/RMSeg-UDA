# RMSeg-UDA: Unsupervised Domain Adpatation for Road Marking Segmentation under Adverse Conditions
Reference implementation about RMSeg-UDA

![image](https://github.com/stu9113611/RMSeg-UDA/blob/main/architecture.png)

This repository contains the reference implementation for RMSeg-UDA, an UDA road marking segmentation training framework.

## Table of Contents
1. [Introduction](#introduction)
2. [Environment](#environment)
3. [Dataset](#dataset)
4. [Training](#training)
5. [Evaluation](#evaluation)

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

## Training

Configuration files in JSON foramt are used in this framework. For practical examples, please check the [configs folder](https://github.com/stu9113611/RMSeg-UDA/tree/main/configs).

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

## Evaluation
