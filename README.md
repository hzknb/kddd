# 基于蒸馏学习的模型转换

## 1. 知识蒸馏

实现从PyTorch向PaddlePaddle跨框架的模型转换. 

### 1.1 教师模型

深度学习框架: PyTorch

教师模型: 
1. ResNet34
2. ResNet50
3. ResNet101


### 1.2 学生模型

深度学习框架: PaddlePaddle

学生模型: 
1. ResNet18


### 1.3 数据集

1. CIFAR10数据集


## 2. 文件结构

### 2.1 前端

web文件包含了测试时要用的文件, 其中`server1.py`为服务端程序, `creat-project-1.html`为前端页面. 

使用步骤: 
1. 打开终端, 进入web文件里, 输入`python server1.py`, 运行`server1.py`代码, 启动服务端; 
2. 双击`creat-project-1.html`, 打开前端页面; 
3. 在页面的输入栏中输入对应数据, 启动跨平台模型转换. 

### 2.2 后端

- `models/`
    - `PaddleX`: 存放PaddlePaddle框架下已经训练好的模型; 
    - `PyTorch`: 存放PyTorch框架下已经训练好的模型. 
- `CrossPlatform/`
  - `DataSet`: 封装了相关数据集, 包括MNIST, CIFAR10等数据集; 
  - `KD/` 
    - `PyTorch_To_Paddle/KD.py`: PyTorch向PaddlePaddle跨框架知识蒸馏代码, 主要求解loss; 
    - `PyTorch_To_PyTorch/KD.py`: PyTorch向PyTorch跨框架知识蒸馏代码, 主要求解loss; 
 