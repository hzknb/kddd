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

# 训练时的实时数据
train_data = []
train_process = []
finish_data = []
img_data = []

app = Flask(__name__)
CORS(app, supports_credentials=True)

def trainKD(T, alpha, epoches, teacher_module, student_module, LOSS, name):
    # 相关设置
    img_data.clear()
    img_data.append("图片参数未获得")
    train_data[0] = "模型导入中"
    finish_data[0] = "未完成训练"
    train_process[0] = f"项目导入中：\n<项目名称:{name}>---<损失函数:{LOSS}>\n<教师模型:{teacher_module}>---<教师框架:{'PyTorch'}>\n<学生模型:{student_module}>---<学生框架:{'PaddlePaddle'}>\n<样本数据:{'CIFAR10'}>---<蒸馏温度:{T}>\n<alpha:{alpha}>"
    
    if not torch.cuda.is_available():
        print("CUDA is not available, loading model to CPU")
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')

    # 1. 教师模型 (ResNet系列)
    if teacher_module == 'ResNet101':
        teacher_model = torchvision.models.resnet101(pretrained=False)
    elif teacher_module == 'ResNet50':
        teacher_model = torchvision.models.resnet50(pretrained=False)
    elif teacher_module == 'ResNet34':
        teacher_model = torchvision.models.resnet34(pretrained=False)
    else:  # 默认ResNet34
        teacher_model = torchvision.models.resnet34(pretrained=False)

    # 修改卷积层以及全连接层
    # 1. 修改卷积层
    teacher_model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
    teacher_model.maxpool = nn.MaxPool2d(1, 1, 0)
    
    # 2. 修改全连接层
    num_features = teacher_model.fc.in_features
    teacher_model.fc = nn.Linear(num_features, 10)
    
    # 加载教师模型参数
    teacher_model.load_state_dict(torch.load(f'../models/PyTorch/{teacher_module}.pth', map_location=device))
    
    # 将教师模型部署在GPU上
    teacher_model.to(device)

    # 设置教师模型为推理模式
    teacher_model.eval()

    # 2. 学生模型
    student_model = resnet18(num_classes=10)
    kd = KD.KD(T=T, alpha=alpha, beta=1-alpha, epoches=epoches)
    
    # 2.1 训练数据集
    train_dataset = CIFAR10.train_dataset
    train_loader = CIFAR10.train_loader
    
    # 2.2 评估数据集
    test_dataset = CIFAR10.test_dataset
    test_loader = CIFAR10.test_loader
    
    # 2.3 统计相关训练数据集以及测试数据集的大小
    train_total_data = 0
    test_total_data = 0
    Init_student_acc = 0
    Init_teacher_acc = 0
    for image, label in train_loader():
        train_total_data += 32
    for image, label in test_loader():
        test_total_data += 32

    # 3. 比较教师模型和未训练学生模型
    T_num_correct = 0
    S_num_correct = 0
    num_total = 0
    student_model.eval()

    for image, label in test_loader():
        # 学生模型
        out = student_model(image)
        pre = paddle.argmax(out, axis=1)
        equal_tensor = paddle.equal(label, pre)
        same_count = int(paddle.sum(equal_tensor).numpy())
        S_num_correct += same_count

        # 教师模型
        image_torch = (torch.from_numpy(image.numpy())).to(device)
        label_torch = (torch.from_numpy(label.numpy())).to(device)
        prediction_torch = teacher_model(image_torch).max(1).indices  # 教师模型预测输出
        T_num_correct += (prediction_torch == label_torch).sum()

        num_total += 32

        # 向前端传入数据, 直观显示项目导入进度
        train_data[0] = "项目导入的进度为：{:.2f}%".format((num_total / test_total_data) * 100)
    
    # 比较教师模型和未训练学生模型的准确率
    Init_student_acc = S_num_correct / num_total * 100
    Init_teacher_acc = T_num_correct / num_total * 100

    # 向前端传入数据, 直观比较教师模型和未训练学生模型的准确率
    img_data.append("教师模型: ")
    img_data.append(round(float(Init_teacher_acc), 2))
    img_data.append("epoch=0: ")
    img_data.append(round(float(Init_student_acc), 2))

    # 提示用户模型导入完成
    train_data[0] = "训练中"
    train_process[0] = "项目导入成功"
    
    # 进入跨框架模型迁移
    optim = paddle.optimizer.Adam(parameters=student_model.parameters())  # 优化器
    for epoch in range(epoches):
        # 训练模式
        student_model.train()
        acc_total_data = 0
        for image, label in train_loader():
            # 教师模型预测
            # 1. 将输入数据转换为PyTorch框架下的数据类型
            image_torch = torch.from_numpy(image.numpy()).to(device)

            # 2. 教师模型预测输出
            prediction_torch = teacher_model(image_torch)

            # 3. 将模型预测输出转换为paddle框架下的数据类型
            y_t = paddle.to_tensor(prediction_torch.cpu().detach().numpy())
            
            # 学生模型预测
            y_s = student_model(image)
            
            # 损失函数
            loss = kd.forward(y_s, y_t, label, epoch, LOSS=LOSS)
            loss.backward()  # 反向传播
            
            # 优化器
            optim.step()  # 更新参数
            optim.clear_grad()  # 梯度清零
            
            acc_total_data += 32

            # 向前端传入数据, 显示模型训练进度
            train_process[0] = "epoch={}的训练进度为：{:.2f}%".format(epoch + 1, (acc_total_data / train_total_data) * 100)
        
        # 提示用户, 一轮训练已经完成
        train_process[0] = "epoch={}训练完成".format(epoch+1)

        # 评测模式
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

            # 向前端传入数据, 显示模型测试进度
            train_process[0] = "epoche={}训练后的模型测试进度为：{:.2f}%".format(epoch+1, (num_total / test_total_data) * 100)

        # 展示每周期模型训练的结果
        if(epoch == 0):
            train_data[0] = "教师模型准确率={:.2f}%".format(Init_teacher_acc)
            img_data.append(f"epoch={epoch+1}:")
            img_data.append(round(float(num_correct / num_total * 100), 2))
        if(len(train_data) < 6):
            train_data.append("\nepoch:{}, 学生模型测试准确率={:.2f}%".format(epoch+1, num_correct / num_total * 100))
            train_data[0] = train_data[0] + "\nepoch:{}, 学生模型测试准确率={:.2f}%".format(epoch+1, num_correct / num_total * 100)
        else:
            train_data[1] = train_data[2]
            train_data[2] = train_data[3]
            train_data[3] = train_data[4]
            train_data[4] = train_data[5]
            train_data[5] = "\nepoche:{}, 学生模型测试准确率={:.2f}%".format(epoch+1, num_correct / num_total * 100)
            train_data[0] = "教师模型准确率={:.2f}%".format(Init_teacher_acc) + train_data[1] + train_data[2] + train_data[3] + train_data[4] + train_data[5]
        if(epoch != 0 and epoch != epoches-1 and epoch == epoches//2):
            img_data.append(f"epoch={epoch + 1}:")
            img_data.append(round(float(num_correct / num_total * 100), 2))

    # 展示跨框架模型迁移之后的效果以及项目设置参数
    if(epoches != 1):
        img_data.append(f"epoch={epoch + 1}:")
        img_data.append(round(float(num_correct / num_total * 100), 2))

    # 保存迁移之后的模型
    paddle.save(student_model.state_dict(), "./static/models/model.pdparams")
    paddle.save(optim.state_dict(), "./static/models/model.pdopt")
    
    train_process[0] = "训练完成,可下载转换后模型"
    train_data[0] = f"训练完成：\n<项目名称:{name}>\n<教师模型:{teacher_module}>---<教师框架:{'PyTorch'}>\n<学生模型:{student_module}>---<学生框架:{'PaddlePaddle'}>\n<损失函数:{LOSS}>-<样本数据:{'CIFAR10'}>-<蒸馏温度:{T}>-<alpha:{alpha}>-<epoch:{epoch+1}>"
    finish_data[0] = "训练完成"
    img_data[0] = "图片参数获取成功"

@app.route("/getdata", methods=["POST"])
def get_sum():
    T = 1
    alpha = 1
    epoches = 1
    teacher_module = 'Teacher1'
    student_module = 'Student'
    loss = 'KL'
    name = 'None'

    # 确定相关传入参数
    print("header {}".format(request.headers))
    print("args ", request.args)
    print("form {}".format(request.form.to_dict()))

    # 获取前端用户输入的参数
    for key, value in request.form.to_dict().items():
        if key == 'T':
            T = int(value)
        elif key == 'alpha':
            alpha = float(value)
        elif key == 'epoch':
            epoches = int(value)
        elif key == 'module_T':
            teacher_module = value
        elif key == 'module_S':
            student_module = value
        elif key == 'loss':
            loss = value
        elif key == 'project_name':
            name = value
    
    # 多线程训练模型
    train = threading.Thread(target=trainKD, args=(T, alpha, epoches, teacher_module, student_module, loss, name))
    train.start()

    return render_template('data1.html')

@app.route('/get_data', methods=['POST', 'GET'])
def get_data():
    return train_data[0]

@app.route('/img', methods=['POST', 'GET'])
def get_img():
    return jsonify(img_data)

@app.route('/train_process', methods=['POST', 'GET'])
def get_process():
    return train_process[0]

@app.route('/button', methods=['POST', 'GET'])
def get_button():
    return finish_data[0]

if __name__ == "__main__":
    train_data.append("未开始训练")
    finish_data.append("未完成训练")
    train_process.append("开始训练")
    img_data.append("图片参数未获得")
    app.config["JSON_AS_ASCII"] = False
    app.run(host="127.0.0.1", port=5050)