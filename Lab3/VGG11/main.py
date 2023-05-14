import torch
from torch import nn
import pandas as pd
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader,random_split,Subset
from PIL import Image
import math

# def seed_target_transform(target:int):
#     folder_name = dataset_test.classes[target]
#     return dataset_train.class_to_idx[folder_name]

def vgg_block(num_convs,in_channels,out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    #拆分成一个又一个元素
    return nn.Sequential(*layers)

conv_arch = ((1,64),(1,128),(2,256),(2,512),(2,512))

def vgg(conv_arch):
    conv_blks = []
    in_channels = 3
    for (num_convs,out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs,in_channels,out_channels))
        in_channels = out_channels

    return nn.Sequential(*conv_blks,nn.Flatten(),
                         nn.Linear(out_channels * 7 * 7,4096),nn.BatchNorm1d(4096),nn.ReLU(),
                         nn.Dropout(0.5),nn.Linear(4096,4096),nn.BatchNorm1d(4096),nn.ReLU(),
                         nn.Dropout(0.5),nn.Linear(4096,12))

class VGGNet(torch.nn.Module):
    def __init__(self,init_weights:bool):
        super(VGGNet,self).__init__()
        self.layer = vgg(conv_arch)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        return self.layer(x)

    def _initialize_weights(self):
        # 继承nn.Module类中的一个方法:self.modules(), 他会返回该网络中的所有modules
        for m in self.modules():
            # isinstance(object, type)：如果指定对象是指定类型，则isinstance()函数返回True
            # 如果是卷积层
            if isinstance(m, nn.Conv2d):
                # uniform_(tensor, a=0, b=1)：服从~U(a,b)均匀分布，进行初始化
                nn.init.xavier_uniform_(m.weight)
                # 如果偏置不是0，将偏置置成0，对偏置进行初始化
                if m.bias is not None:
                    # constant_(tensor, val)：初始化整个矩阵为常数val
                    nn.init.constant_(m.bias, 0)
            # 如果是全连接层
            elif isinstance(m, nn.Linear):
                # 正态分布初始化
                nn.init.xavier_uniform_(m.weight)
                # 将所有偏执置为0
                nn.init.constant_(m.bias, 0)

    def predict(self, x,batchsize=16):  # 该函数用在测试集过程，因此只有前向传播，没有什么
        with torch.no_grad():
            res = []  # 将测试的结果都汇集到这个列表中
            for i in range(math.ceil(x.shape[0]/batchsize)):
                if i + 1 * batchsize < x.shape[0]:
                    x_batch = x[i * batchsize:(i + 1)*batchsize].to(device)
                else:
                    x_batch = x[i * batchsize:].to(device)
                x_batch = self.layer(x_batch)
                _, predicted = torch.max(x_batch, dim=1)
                for j in range(predicted.shape[0]):
                   res.append(full_dataset.classes[predicted[j].item()])
        return res

size = 224
batch_size = 32
device = torch.device("cuda:0")

data_transform = {
        # Compose()：将多个transforms的操作整合在一起
        # 训练
        "train": transforms.Compose([
            # RandomResizedCrop(224)：将给定图像随机裁剪为不同的大小和宽高比，然后缩放所裁剪得到的图像为给定大小
            transforms.RandomResizedCrop(224),
            # ToTensor()：数据转化为Tensor格式
            transforms.ToTensor(),
            # Normalize()：将图像的像素值归一化到[-1,1]之间，使模型更容易收敛
            transforms.Normalize([0.329, 0.289, 0.207], [0.091, 0.094, 0.104])]),
        # 验证
        "val": transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.329, 0.289, 0.207], [0.091, 0.094, 0.104])])}


full_dataset= ImageFolder('./train')

train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_indices, test_indices = random_split(list(range(len(full_dataset))), [train_size, test_size])

train_dataset = Subset(full_dataset, train_indices)
test_dataset = Subset(full_dataset, test_indices)

# 对数据集使用不同的 transform
train_dataset.dataset.transform = data_transform['val']
test_dataset.dataset.transform = data_transform['val']
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


model = VGGNet(init_weights=True)
device = torch.device("cuda:0")
model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.0005)

#定义训练函数
def train(epoch):
    running_loss = 0.0
    for batch_idx,data in enumerate(train_loader,0):
        inputs,labels = data
        inputs,labels = inputs.to(device),labels.to(device)
        optimizer.zero_grad()

        #forward + backward + update
        outputs = model(inputs)
        loss = criterion(outputs,labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 10 == 9:
            print('[%d,%5d] loss : %.3f' % (epoch + 1,batch_idx + 1,running_loss / 10))
            running_loss = 0.0

def test(test_loader,*kwrag):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images,labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _,predicted = torch.max(outputs,dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy on %s set: %lf %%' % (*kwrag,100 * correct / total))
    return correct / total

# def predict(test_loader,*kwrag):
#     res = []
#     with torch.no_grad():
#         for image in test_loader:
#             image = image.to(device)
#             output = model(image)
#             _,predicted = torch.max(output,dim=1)
#             res.append(predicted.item())
#     return res

def readData(filename):
    data = pd.read_csv(filename)
    cnt = data.shape[0]
    images_tensor = []
    for i in range(cnt):
        image = Image.open('./test/Sugar beet/' + data['file'][i]).convert('RGB')
        image = data_transform['val'](image)
        images_tensor.append(image)
    stacked_tensor = torch.stack(images_tensor, dim=0)
    return stacked_tensor,data

if __name__ == '__main__':
    # checkpoint = torch.load('./params/' + '4.pth')
    # model.load_state_dict(checkpoint['net'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
    # test(test_loader, 'test')

    cnt = 0
    index = 0
    for epoch in range(1,30):
        train(epoch)
        tmp = test(test_loader, 'val-test')
        if tmp > cnt:
            cnt = tmp
            index = epoch
            state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
            torch.save(state, 'C:/Users/86156/Desktop/params/params1' + str(epoch) + '.pth')
    torch.save(state, 'C:/Users/86156/Desktop/params/params2' + str(epoch) + '.pth')
    print(index)

    kaggle_input,kaggle_data = readData('./sample_submission.csv')
    kaggle_predict = model.predict(kaggle_input)

    outputs = pd.DataFrame({'file': kaggle_data.file, 'species': kaggle_predict})
    outputs.to_csv(r"predicted.csv", index=False)  # index=False 代表不保存索引




