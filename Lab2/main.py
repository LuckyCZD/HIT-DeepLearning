import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms

class AlexNet(nn.Module):
    def __init__(self, num_classes=101):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def train(model, optimizer, criterion, train_loader, valid_loader, writer, num_epochs):

    # 将模型设置为训练模式
    model.train()

    # 定义最佳验证集准确率和最佳模型参数
    best_valid_acc = 0.0
    best_model_params = model.state_dict()

    for epoch in range(num_epochs):
        train_loss = 0.0
        train_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0

        # 训练模型
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_acc += torch.sum(preds == labels.data)

        # 在验证集上测试模型
        with torch.no_grad():
            model.eval()
            for inputs, labels in valid_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                valid_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                valid_acc += torch.sum(preds == labels.data)

        # 计算平均损失和准确率
        train_loss /= len(train_loader.dataset)
        train_acc /= len(train_loader.dataset)
        valid_loss /= len(valid_loader.dataset)
        valid_acc /= len(valid_loader.dataset)

        # 记录训练和验证集的损失和准确率
        writer.add_scalar('train_loss', train_loss, epoch + 1)
        writer.add_scalar('train_acc', train_acc, epoch + 1)
        writer.add_scalar('valid_loss', valid_loss, epoch + 1)
        writer.add_scalar('valid_acc', valid_acc, epoch + 1)

        print('Epoch [{}/{}], Train Loss: {:.4f}, Train Acc: {:.4f}, Valid Loss: {:.4f}, Valid Acc: {:.4f}'
              .format(epoch+1, num_epochs, train_loss, train_acc, valid_loss, valid_acc))

        # 如果当前模型在验证集上的准确率更好，则保存模型参数
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_model_params = model.state_dict()

    # 加载最佳模型参数
    model.load_state_dict(best_model_params)

    # 保存模型
    torch.save(model.state_dict(), 'alexnet.pth')


def test(test_loader, criterion, writer):
    new_model = AlexNet(num_classes=101)
    # 加载保存的模型参数
    checkpoint = torch.load('alexnet.pth')
    new_model.load_state_dict(checkpoint)

    # 设置模型为评估模式
    new_model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = new_model(data)
            test_loss += criterion(output, target).item()  # 将一批的损失相加
            pred = output.argmax(dim=1, keepdim=True)  # 找到概率最大的下标
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), test_accuracy))
    writer.add_scalar('Testing Loss', test_loss)
    writer.add_scalar('Testing Accuracy', test_accuracy)


def get_data_loaders(data_dir, batch_size):
    # 定义数据预处理方式，包括裁剪、缩放和标准化
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 加载数据集
    full_dataset = datasets.ImageFolder(os.path.join(data_dir), transform=transform)

    # 将数据集按照数量划分为训练集、验证集和测试集
    num_samples = len(full_dataset)
    num_train = int(num_samples * 0.8)
    num_valid = int(num_samples * 0.1)
    num_test = num_samples - num_train - num_valid

    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [num_train, num_valid, num_test])
    print(len(train_dataset), len(valid_dataset), len(test_dataset))
    # 创建DataLoader对象
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, valid_loader, test_loader


if __name__ == '__main__':

    # 设置超参数
    num_epochs = 30
    batch_size = 64
    learning_rate = 0.0001

    # 定义tensorboard日志文件夹路径
    log_dir = 'runs/alexnet_caltech101'

    # 初始化tensorboard的SummaryWriter
    writer = SummaryWriter(log_dir=log_dir)

    # 获取数据加载器
    train_loader, valid_loader, test_loader = get_data_loaders('./caltech-101/101_ObjectCategories', batch_size)

    # 创建模型
    model = AlexNet(num_classes=101)

    # 将模型移动到GPU上进行训练
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    train(model, optimizer, criterion, train_loader, valid_loader, writer, num_epochs)

    # 测试模型
    test(test_loader, criterion, writer)

    # 关闭tensorboard的SummaryWriter
    writer.close()





