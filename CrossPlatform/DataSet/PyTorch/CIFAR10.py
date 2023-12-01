import torch
import torchvision
import torchvision.datasets
import torch.utils.data
import torchvision.transforms as transforms

# 数据预处理
transform = transforms.Compose([
                                transforms.ToTensor()
                                ])

# 加载CIFAR10数据集
train_dataset = torchvision.datasets.CIFAR10(root='./CIFAR10', 
                                             train=True,
                                             download=True, 
                                             transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, 
                                           batch_size=32,
                                           shuffle=True)

test_dataset = torchvision.datasets.CIFAR10(root='./CIFAR10',
                                            train=False,
                                            download=True, 
                                            transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, 
                                          batch_size=32,
                                          shuffle=False)
