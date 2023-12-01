# 导入相关模块
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import time
import threading
import sys

import numpy as np

import paddle
import paddle.nn
import paddle.nn.functional as F
import paddle.optimizer
import paddle.metric
from paddle.metric import Accuracy
from paddle.vision.models import resnet18

sys.path.append('..')
import CrossPlatform.KD.PyTorch_To_Paddle.KD as KD
import CrossPlatform.DataSet.PaddleX.CIFAR10 as CIFAR10

import torch
import torchvision.models
import torch.nn as nn

paddle.set_device("gpu:0")

if not torch.cuda.is_available():
    print("CUDA is not available, loading model to CPU")
    device = torch.device('cpu')
else:
    device = torch.device('cuda:0')

student_model = resnet18(num_classes=10)

layer_state_dict = paddle.load("model_params/model.pdparams")
student_model.set_state_dict(layer_state_dict)

# 2.1 训练数据集
train_dataset = CIFAR10.train_dataset
train_loader = CIFAR10.train_loader

# 2.2 评估数据集
test_dataset = CIFAR10.test_dataset
test_loader = CIFAR10.test_loader

train_total_data = 0
test_total_data = 0
Init_student_acc = 0
Init_teacher_acc = 0

for image, label in train_loader():
    train_total_data += 32
for image, label in test_loader():
    test_total_data += 32

num_correct = 0
num_total = 0
student_model.eval()
for image, label in test_loader():
    out = student_model(image)
    pre = paddle.argmax(out, axis=1)

    # 1. 比较label和pre张量中对应位置的元素是否相等
    equal_tensor = paddle.equal(label, pre)

    # 2. 计算True的数量，即相同数的个数
    same_count = int(paddle.sum(equal_tensor).numpy())
    num_correct += same_count
    num_total += 32

    #显示模型测试进度
    print("模型测试进度为：{:.2f}%".format((num_total / test_total_data) * 100))

print("模型在测试集上的准确率为：{:.2f}%".format((num_correct / num_total) * 100))



