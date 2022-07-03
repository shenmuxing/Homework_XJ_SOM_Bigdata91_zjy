# 项目说明

## 声明

部分代码来自于[此notebook](https://github.com/denizhankara/PPG-DaLiA/blob/main/notebooks/Sample-data-analysis.ipynb)

## 数据来源

数据来源于[UCI PPG Dataset](https://archive.ics.uci.edu/ml/datasets/PPG-DaLiA)

本仓库不提供数据本身，需要将数据自行下载并放在data文件夹下。

## 项目目的

本数据是15位受试者在36小时内进行各种活动得到的数据集，试图通过各种检测设备的特征预测受试者实际心率水平。

## 项目思路

1. 数据预处理：详见文件`Project->data->1. 数据预处理`。主要是将20G的数据进行读取，提取最原始的特征，构建一个DataFrame