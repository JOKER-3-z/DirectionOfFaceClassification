import os
import torch
import numpy as np
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader,Dataset

#image是头部旋转图像，这里label是头部旋转类别
class PoseDataset(Dataset):
    def __init__(self,image,label,transform=None):
        self.transform = transform
        self.images = image
        self.labels = label

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].reshape(48,32,1)
        label = self.labels[idx]
        # 如果有转换方式，进行转换
        if self.transform is not None:
            image = self.transform(image)
        else:
            # 归一化，转为torch
            image = torch.torch.tensor(image/255,dtype=torch.float)
        return image,label

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    ## 配置其它训练超参数 batch_size：批数据大小,num_workers：线程,learning rate：学习率,以及epochs:训练轮次
    batch_size = 256
    num_workers = 1
    lr = 1e-4
    epochs = 20

    ##数据读入与加载
    # Dateset: 继承类定制特殊需求
    # DataLoader: 数据的读入

    # 数据读入需要变换使用torchvision,将数据转为Tensor


    image_size = 28
    data_transform = transforms.Compose([
        transforms.ToPILImage(),  # 取决于后续数据读入方式，使用内部数据集时不需要
        transforms.Resize(48,32),  # 转为指定size
        transforms.ToTensor()  # 转为tensor，方便输入网络
    ])

    # 读数据
    train_df = pd.read_csv("./FashionMNIST/fashion-mnist_train.csv")
    test_df = pd.read_csv("./FashionMNIST/fashion-mnist_test.csv")

    # 数据格式转换
    train_data = FMDataset(train_df,data_transform)
    test_data = FMDataset(test_df,data_transform)

    # 定义dataloader: dataset,batch_size,shuffer:随机打乱取数据,num_workers,drop_last:丢弃最后一个batch，防止最后一批次不足一批，pin_memory:将数据放到锁页内存，那空间换时间
    train_loader=DataLoader(train_data,batch_size=batch_size,shuffle=True,num_workers=num_workers,drop_last=True)
    test_loader=DataLoader(test_data,batch_size=batch_size,shuffle=False,num_workers=num_workers)

    # 数据可视化
    #import matplotlib.pyplot as plt
    # 从train_loader读数据
    #image,label = next(iter(train_loader))
    #print(image.shape,label.shape)
    #plt.imshow(image[0][0],cmap='gray')
    #plt.show()