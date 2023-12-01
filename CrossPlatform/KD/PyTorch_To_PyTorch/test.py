import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet50, resnet18

# 1. 准备训练集
# 设置随机种子
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 使用cuda进行加速卷积运算
torch.backends.cudnn.benchmark = True

# 载入数据集
# 1> MINST数据集
classes = ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10')

# 1.1 教师模型数据集
train_dataset = torchvision.datasets.MNIST(root="./MINST",
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)
train_dataloder = DataLoader(train_dataset,
                             batch_size=32,
                             shuffle=True)
test_dataset = torchvision.datasets.MNIST(root="./MINST",
                                          train=False,
                                          transform=transforms.ToTensor(),
                                          download=True)
test_dataloder = DataLoader(test_dataset,
                            batch_size=32,
                            shuffle=True)

# 1.2 学生模型数据集 (使用相对比较少的数据进行测试和验证)
subset_indices = range(0, int(0.2 * len(train_dataset)))
subset_dataset = torch.utils.data.Subset(train_dataset,
                                         indices=subset_indices)
subset_dataloader = torch.utils.data.DataLoader(subset_dataset,
                                                batch_size=32,
                                                shuffle=True)
subset_testset = torch.utils.data.Subset(test_dataset,
                                         indices=subset_indices)
subset_testloader = torch.utils.data.DataLoader(subset_testset,
                                                batch_size=32,
                                                shuffle=True)


# # 2> CIFAR10数据集
# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # 数据预处理
# classes = ('plane', 'car', 'bird', 'cat', 'deer', 
#            'dog', 'frog', 'horse', 'ship', 'truck')  # 十分类

# # 2.1 教师模型数据集
# train_dataset = torchvision.datasets.CIFAR10(root='./data',
#                                              train=True,
#                                              download=True,
#                                              transform=transform)
# train_dataloader = torch.utils.data.DataLoader(train_dataset,
#                                                batch_size=32,
#                                                shuffle=True)
# test_dataset = torchvision.datasets.CIFAR10(root='./data',
#                                             train=False,
#                                             download=True,
#                                             transform=transform)
# test_dataloader = torch.utils.data.DataLoader(test_dataset,
#                                               batch_size=32,
#                                               shuffle=True)

# # 2.2 学生模型数据集 (使用相对比较少的数据进行测试和验证)
# subset_indices = range(0, int(0.2 * len(train_dataset)))
# subset_dataset = torch.utils.data.Subset(train_dataset,
#                                          indices=subset_indices)
# subset_dataloader = torch.utils.data.DataLoader(subset_dataset,
#                                                 batch_size=32,
#                                                 shuffle=True)
# subset_testset = torch.utils.data.Subset(test_dataset,
#                                          indices=subset_indices)
# subset_testloader = torch.utils.data.DataLoader(subset_testset,
#                                                 batch_size=32,
#                                                 shuffle=True)


# 2. 搭建教师神经网络
# 2.1> 演示小型神经网络
class Teacher_model(nn.Module):
    def __init__(self, in_channels=1, num_class=10):
        super(Teacher_model, self).__init__()
        self.fc1 = nn.Linear(784, 1200)
        self.fc2 = nn.Linear(1200, 1200)
        self.fc3 = nn.Linear(1200, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
 
model = Teacher_model()
model=model.to(device)

# # 2.2> ResNet网络
# model = resnet50(pretrained=True)
# model.fc = nn.Linear(512, len(classes))
# model = model.to(device)

# 损失函数和优化器的设计
loss_function = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(),
                         lr=0.0001)

teacher_model = model


# 4. 搭建学生网络
model = resnet18(pretrained=True)
model.fc = nn.Linear(512, len(classes))
model = model.to(device)
loss_function = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=0.0001)


# 5. 学生网络的训练和预测结果
print("---学生网络---")
epoches = 6
for epoch in range(epoches):
    model.train()
    for image, label in train_dataloder:
        image, label = image.to(device), label.to(device)
        optim.zero_grad()
        out = model(image)
        loss = loss_function(out, label)
        loss.backward()
        optim.step()

    model.eval()
    num_correct = 0
    num_samples = 0
    with torch.no_grad():
        for image, label in test_dataloder:
            image, label = image.to(device), label.to(device)
            out = model(image)
            pre = out.max(1).indices
            num_correct += (pre == label).sum()
            num_samples += pre.size(0)
        acc = (num_correct / num_samples).item()

    model.train()
    print("epoches:{},accurate={}".format(epoch, acc))


# 6. 知识蒸馏
#   1> PyTorch to PyTorch
# 6.1 将教师网络设置为推理模式 (ResNet50)
teacher_model.eval()
# 6.2 部署学生网络进行训练 (ResNet18)
model = resnet18(pretrained=True)
model.fc = nn.Linear(512, len(classes))
model = model.to(device)
T = 7
hard_loss = nn.CrossEntropyLoss()
alpha = 0.3
soft_loss = nn.KLDivLoss(reduction="batchmean")
optim = torch.optim.Adam(model.parameters(), lr=0.0001)

print("---知识蒸馏---")
epoches = 5
for epoch in range(epoches):
    # 模型训练
    model.train()
    for image, label in subset_dataloader:
        image, label = image.to(device), label.to(device)
        with torch.no_grad():
            teacher_output = teacher_model(image)
        optim.zero_grad()
        out = model(image)
        loss = hard_loss(out, label)
        ditillation_loss = soft_loss(F.softmax(out / T, dim=1), F.softmax(teacher_output / T, dim=1))
        loss_all = loss * alpha + ditillation_loss * (1 - alpha)
        loss_all.backward()
        optim.step()

    # 模型评估
    model.eval()
    num_correct = 0
    num_samples = 0
    with torch.no_grad():
        for image, label in subset_testloader:
            image, label = image.to(device), label.to(device)
            out = model(image)
            pre = out.max(1).indices
            num_correct += (pre == label).sum()
            num_samples += pre.size(0)
        acc = (num_correct / num_samples).item()

    print("epoches:{},accurate={}".format(epoch, acc))

# 2> PyTorch蒸馏到PaddlePaddle
# # 6.1 将教师网络设置为推理模式
# teacher_model.eval()
# # 6.2 部署学生网络进行训练
# model = resnet18(pretrained=True)
# model.fc = nn.Linear(512, len(classes))
# model = model.to(device)
# # 蒸馏温度
# T = 7
# hard_loss = nn.CrossEntropyLoss()
# alpha = 0.3
# soft_loss = nn.functional.kl_div(reduction="batchmean")
# optim = optim.Adam(model.parameters(),
#                          lr=0.0001)
# print("---知识蒸馏---")
# epoches = 5
# for epoch in range(epoches):
#     model.train()  # 训练模式
#     for image, label in subset_dataloader:
#         image, label = image.to(device), label.to(device)
#         with torch.no_grad():
#             teacher_output = teacher_model(image)
#             teacher_output = torch.from_numpy(teacher_output.numpy())
#         optim.zero_grad()
#         out = model(image)
#         loss = hard_loss(out, label)
#         ditillation_loss = soft_loss(F.softmax(out / T, dim=1), F.softmax(teacher_output / T, dim=1))
#         loss_all = loss * alpha + ditillation_loss * (1 - alpha)
#         loss_all.backward()
#         optim.step()

#     model.eval()  # 评估模式
#     num_correct = 0
#     num_samples = 0
#     with torch.no_grad():
#         for image, label in subset_testloader:
#             image, label = image.to(device), label.to(device)
#             out = model(image)
#             pre = out.max(1).indices
#             num_correct += (pre == label).sum()
#             num_samples += pre.size(0)
#         acc = (num_correct / num_samples).item()

#     model.train()
#     print("epoches:{},accurate={}".format(epoch, acc))
