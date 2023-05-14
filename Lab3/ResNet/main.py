import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim as optim
import pandas as pd
from PIL import Image
import csv
from torch.utils.tensorboard import SummaryWriter


# 定义ResNet18模型
class ResNet18(nn.Module):
    def __init__(self, num_classes, block_type='basic', use_gpu=True):
        super(ResNet18, self).__init__()
        self.use_gpu = use_gpu
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block_type, 64, 2)
        self.layer2 = self._make_layer(block_type, 128, 2, stride=2)
        self.layer3 = self._make_layer(block_type, 256, 2, stride=2)
        self.layer4 = self._make_layer(block_type, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block_type, out_channels, num_blocks, stride=1):
        if block_type == 'basic':
            block = BasicBlock
        else:
            raise ValueError('Unsupported block type: %s' % block_type)
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )
        layers = [block(self.in_channels, out_channels, stride, downsample, self.use_gpu)]
        self.in_channels = out_channels * block.expansion
        for i in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels, use_gpu=self.use_gpu))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# 定义BasicBlock模块
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, use_gpu=True):
        super(BasicBlock, self).__init__()
        self.use_gpu = use_gpu
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        # 添加 SE block
        # self.stride = stride
        # self.se = SEBlock(out_channels, reduction_ratio=16)

    def forward(self, x):
        identity = x
        x = self.conv1(x.cuda())
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.downsample is not None:
            identity = self.downsample(identity.cuda())
        x += identity
        x = F.relu(x)
        return x


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio)
        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y


def train(model, loader, criterion, optimizer, use_gpu):
    # 切换到训练模式
    model.train()

    # 循环遍历数据集
    total_loss = 0.0
    correct = 0
    total = 0
    for i, (inputs, targets) in enumerate(loader):
        if use_gpu:
            inputs = inputs.cuda()
            targets = targets.cuda()

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计数据
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().item()

        # 输出统计结果
        if (i + 1) % 10 == 0:
            print('Train [{}/{}], Loss: {:.4f}, Acc: {:.4f}'.format(
                i + 1, len(loader), total_loss / (i + 1), 100.0 * correct / total))

    # 返回平均损失和准确率
    return total_loss / len(loader), 100.0 * correct / total


def test(model, loader, criterion, use_gpu):
    # 切换到评估模式
    model.eval()

    # 循环遍历数据集
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader):
            if use_gpu:
                inputs = inputs.cuda()
                targets = targets.cuda()

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # 统计数据
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum().item()

    # 返回平均损失和准确率
    return total_loss / len(loader), 100.0 * correct / total


def test_csv(model, loader, use_gpu):
    # 切换到评估模式
    model.eval()
    name = [['Black-grass'], ['Charlock'], ['Cleavers'], ['Common Chickweed'], ['Common wheat'], ['Fat Hen'],
            ['Loose Silky-bent'], ['Maize'], ['Scentless Mayweed'], ['Shepherds Purse'], ['Small-flowered Cranesbill'],
            ['Sugar beet']]
    with open('predict3.csv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['species'])
        with torch.no_grad():
            for i, inputs in enumerate(loader):
                if use_gpu:
                    inputs = inputs.cuda()

                # 前向传播
                outputs = model(inputs)
                index = torch.argmax(outputs)
                writer.writerow(name[int(index)])
                # print(name[int(index)])


# 定义数据预处理方法
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.329, 0.289, 0.207], [0.091, 0.094, 0.104])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.329, 0.289, 0.207], [0.091, 0.094, 0.104])
    ]),
}

def get_data_loaders():

    # 加载数据集
    data_dir = './plant-seedlings-classification'
    full_dataset = datasets.ImageFolder(root=data_dir + '/train', transform=data_transforms['train'])

    # 将数据集按照数量划分为训练集、验证集
    num_samples = len(full_dataset)
    num_train = int(num_samples * 0.8)
    num_valid = num_samples - num_train

    train_dataset, valid_dataset = torch.utils.data.random_split(full_dataset, [num_train, num_valid])
    print(len(train_dataset), len(valid_dataset))

    # 创建数据加载器
    batch_size = 64
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, valid_loader

def readData(filename):
    data = pd.read_csv(filename)
    cnt = data.shape[0]
    images_tensor = []
    for i in range(cnt):
        img = Image.open('./plant-seedlings-classification/test/' + data['file'][i]).convert('RGB')
        img = data_transforms['val'](img)
        images_tensor.append(img)
    stacked_tensor = torch.stack(images_tensor, dim=0)
    test_loader = torch.utils.data.DataLoader(stacked_tensor, batch_size=1, shuffle=False, num_workers=4)
    return test_loader, data


if __name__ == '__main__':

    # # 创建模型和优化器
    # use_gpu = True  # 是否使用GPU
    # num_classes = 12  # 数据集中的类别数
    # model = ResNet18(num_classes=num_classes, use_gpu=use_gpu)
    # if use_gpu:
    #     model.cuda()
    # # optimizer = optim.Adam(model.parameters(), lr=0.001)
    # optimizer = optim.SGD(model.parameters(), lr=0.001)
    # # 定义损失函数
    # criterion = nn.CrossEntropyLoss()
    #
    # # 定义训练参数
    # num_epochs = 30
    #
    # # 定义tensorboard日志文件夹路径
    # log_dir = './resnet'
    #
    # # 初始化tensorboard的SummaryWriter
    # writer = SummaryWriter(log_dir=log_dir)
    #
    # train_loader, val_loader = get_data_loaders()
    #
    # # 循环遍历多个 epoch
    # for epoch in range(num_epochs):
    #     # 训练模型
    #     train_loss, train_acc = train(model, train_loader, criterion, optimizer, use_gpu)
    #
    #     # 在测试集上评估模型
    #     val_loss, val_acc = test(model, val_loader, criterion, use_gpu)
    #
    #     # 输出统计结果
    #     print('Epoch [{}/{}], Train Loss: {:.4f}, Train Acc: {:.4f}%, Val Loss: {:.4f}, Val Acc: {:.4f}%'.format(
    #         epoch + 1, num_epochs, train_loss, train_acc, val_loss, val_acc))
    #
    #     # 记录训练和验证集的损失和准确率
    #     writer.add_scalar('train_loss', train_loss, epoch + 1)
    #     writer.add_scalar('train_acc', train_acc, epoch + 1)
    #     writer.add_scalar('valid_loss', val_loss, epoch + 1)
    #     writer.add_scalar('valid_acc', val_acc, epoch + 1)
    #
    #
    # # 保存模型和优化器的参数
    # torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),}, 'ResNet+SGD.pth')
    #
    # # 关闭tensorboard的SummaryWriter
    # writer.close()


    # 下面是生成测试集的结果csv的代码
    use_gpu = True  # 是否使用GPU
    num_classes = 12  # 数据集中的类别数
    model = ResNet18(num_classes=num_classes, use_gpu=use_gpu)
    if use_gpu:
        model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 加载模型和优化器的参数
    checkpoint = torch.load('ResNet+SEBlock.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    test_loader, data = readData("./plant-seedlings-classification/sample_submission.csv")

    test_csv(model, test_loader, use_gpu)








