import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Subset

classes = ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10')

# 1. 老师模型
# 1.1 训练数据集
train_dataset = torchvision.datasets.MNIST(root="./MINST",
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)
train_dataloder = DataLoader(train_dataset,
                             batch_size=32,
                             shuffle=True)

# 1.2 测试数据集
test_dataset = torchvision.datasets.MNIST(root="./MINST",
                                          train=False,
                                          transform=transforms.ToTensor(),
                                          download=True)
test_dataloder = DataLoader(test_dataset,
                            batch_size=32,
                            shuffle=True)

# 2. 学生模型
subset_train_num = range(0, int(0.2 * len(train_dataset)))  # 20%的数据集
subset_test_num = range(0, int(0.2 * len(test_dataset)))

# 2.1 训练数据集
subset_train_dataset = Subset(train_dataset,
                              indices=subset_train_num)
subset_train_dataloader = DataLoader(subset_train_dataset,
                                     batch_size=32,
                                     shuffle=True)

# 2.2 测试数据集
subset_test_dataset = Subset(test_dataset, 
                              indices=subset_test_num)
subset_test_dataloader = DataLoader(subset_test_dataset, 
                                    batch_size=32, 
                                    shuffle=True)
