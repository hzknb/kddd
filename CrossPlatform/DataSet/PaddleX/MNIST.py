import paddle.io
from paddle.vision import datasets, transforms

transform = transforms.Compose([
    transforms.Resize((28, 28)),  # 将图像大小调整为28*28像素
    transforms.ToTensor(),  # 将图像转换为张量
    transforms.Normalize(mean=[0.5], std=[0.5], data_format='CHW'),  # 对图像进行归一化处理
])

# 使用transform对数据集进行归一化
print("download training data and load training data")

# 1. 训练数据集
train_dataset = datasets.MNIST(mode='train', 
                               transform=transform)

# 2. 测试数据集
test_dataset = datasets.MNIST(mode='test', 
                              transform=transform)
