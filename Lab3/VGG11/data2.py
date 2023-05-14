 import torch
from torchvision.datasets import ImageFolder
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
import numpy as np
import matplotlib.pyplot as plt

class ObjectDataset(Dataset):
    def __init__(self,data_path="./test",size=224):
        self.data_path = data_path
        self.size = size
        self.category_name = sorted(os.listdir(data_path))
        self.label_category = {self.category_name[i]: i for i in range(len(self.category_name))}
        self.sample_numlist = []
        for image_category in self.category_name:
            images_path = os.path.join(self.data_path, image_category)
            self.sample_numlist.append(len(os.listdir(images_path)))
        self.transform_image = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),  # 转化成张量，从 224 * 224 * 3 转化为 3 * 224 * 224
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        sum_len = 0
        for i in self.sample_numlist:
            sum_len = sum_len + i
        return sum_len

    def __getitem__(self, idx):
        for i in range(len(self.sample_numlist)):
            if idx >= self.sample_numlist[i]:
                idx = idx - self.sample_numlist[i]
            else:
                #对应的图像所属的文件夹位置
                images_path = os.path.join(self.data_path,self.category_name[i])
                #对应的图像名字
                image_path = sorted(os.listdir(images_path))[idx]
                image = cv2.imread(os.path.join(images_path, image_path))
                #获得对应的数据，返回对应值
                image = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_AREA)
                label = self.label_category[self.category_name[i]]
                image = self.transform_image(image)
                label = torch.tensor(label)

                return image,label

size = 224
transform = transforms.Compose([
    transforms.Resize([size, size]),
    transforms.ToTensor(), #转化成张量，从 224 * 224 * 3 转化为 3 * 224 * 224
])



#计算整个数据集的均值和方差
def getStat(train_data):
    print('计算均值和方差：')
    print(len(train_data))
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=4,
        pin_memory=True) #使用锁页内存，加快速度，数据集比较小
    mean = torch.zeros(3)
    std = torch.zeros(3)
    #不同的维数做相加
    for X, _ in train_loader:
        for d in range(3):
            X = np.array(X)
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    #除个数
    mean.div_(len(train_data))
    std.div_(len(train_data))
    return list(mean.numpy()), list(std.numpy())


if __name__ == '__main__':
    train_dataset = ImageFolder(root=r'./train', transform=transform)
    test_dataset = ImageFolder(root=r'./test', transform=transform)
    dataset = train_dataset + test_dataset
    print(len(dataset))
    dataloader = DataLoader(dataset=dataset, batch_size=128)
    # print(getStat(dataset))
    figure = plt.figure()
    num_of_images = 60
    for imgs, targets in dataloader:
        break

    for index in range(num_of_images):
        plt.subplot(6, 10, index + 1)
        plt.axis('off')
        img = imgs[index, ...]
        plt.imshow(img.numpy().T, cmap='gray_r')
    figure.subplots_adjust(hspace=0, wspace=0.05)
    figure.suptitle('test 2*2 axes')
    # 设置整个图像的边框颜色和线宽
    plt.show()


