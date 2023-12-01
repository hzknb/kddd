import paddle
from paddle.vision import transforms
import paddle.io

# transform
transform = transforms.Compose([
    transforms.ToTensor()
])

# 训练数据集
train_dataset = paddle.vision.datasets.Cifar10(mode='train', 
                                               transform=transform)
train_loader = paddle.io.DataLoader(train_dataset, 
                                    batch_size=32, 
                                    shuffle=True)

# 测试数据集
test_dataset = paddle.vision.datasets.Cifar10(mode='test', 
                                              transform=transform)
test_loader = paddle.io.DataLoader(test_dataset,
                                   batch_size=32, 
                                   shuffle=True)
