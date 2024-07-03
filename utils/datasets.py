import os
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from tqdm import tqdm
class CustomImageDataset(Dataset):
    def __init__(self, data_dir, classes,transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = classes
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.img_paths = self.get_image_paths()

    def get_image_paths(self):#/dataset/xxclass/xx.jpg
        img_paths = []
        for cls in self.classes:
            class_dir = os.path.join(self.data_dir, cls)
            class_idx = self.class_to_idx[cls]
            for filename in os.listdir(class_dir):
                img_path = os.path.join(class_dir, filename)
                img_paths.append(
                        (
                            img_path, class_idx
                        )
                    )
        return img_paths
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self.img_paths))
            ls = self.img_paths[start:stop]
            res=[]
            for imgpath in ls:
                img_path, class_idx=imgpath
                image = Image.open(img_path).convert('RGB')
        
                if self.transform is not None:
                    image = self.transform(image)
                res.append((image, class_idx))
            return res
        else:
            img_path, class_idx = self.img_paths[idx]
            image = Image.open(img_path).convert('RGB')
            if self.transform is not None:
                image = self.transform(image)
            return image, class_idx
            
from torchvision.transforms import ToTensor#用于把图片转化为张量
import numpy as np#用于将张量转化为数组，进行除法
from torchvision.datasets import ImageFolder#用于导入图片数据集
def get_mean_std(dir):
    means = [0,0,0]
    std = [0,0,0]#初始化均值和方差
    transform=ToTensor()#可将图片类型转化为张量，并把0~255的像素值缩小到0~1之间
    dataset=ImageFolder(dir,transform=transform)#导入数据集的图片，并且转化为张量
    num_imgs=len(dataset)#获取数据集的图片数量
    for img,a in tqdm(dataset,desc='get_mean_std'):#遍历数据集的张量和标签
        for i in range(3):#遍历图片的RGB三通道
            # 计算每一个通道的均值和标准差
            means[i] += img[i, :, :].mean()
            std[i] += img[i, :, :].std()
    mean=np.array(means)/num_imgs
    std=np.array(std)/num_imgs#要使数据集归一化，均值和方差需除以总图片数量
    return mean,std #打印出结果


def get_transform(chance='train',resize_size=337,mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    data_transforms = {
        'train': transforms.Compose([
            # transforms.RandomResizedCrop(resize_size, scale=(0.6, 1.0), ratio=(0.75, 1.33), interpolation=transforms.InterpolationMode.BILINEAR),
            # transforms.RandomResizedCrop(112,scale=(0.5, 1.0), ratio=(3./4., 4./3.)),
            # transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomGrayscale(p=0.4),  # 10%的概率将图像转换为灰度

            # transforms.RandomAffine(degrees, translate=None, scale=None, shear=None),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]),
        'val': transforms.Compose([
            transforms.Resize(resize_size),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]),
        "null": transforms.Compose([
            transforms.Resize(resize_size),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
    }
    return data_transforms[chance]
import os,shutil
from tqdm import tqdm
def split_data(
        data_path='data/lfw',
        tartget_dir='data/lfw_split',
        dataset_sep=[0.8,0.1,0.1],
        datasize=500
    ):
    if not os.path.exists(tartget_dir):
        os.makedirs(tartget_dir)
    shutil.rmtree(tartget_dir)

    total_data = [os.path.join(data_path, path) for path in os.listdir(data_path)][:datasize]
    train_num=int(len(total_data)*dataset_sep[0])
    val_num=int(len(total_data)*dataset_sep[1])+train_num
    test_num=int(len(total_data)*dataset_sep[2])+val_num
    for i in tqdm(range(len(total_data))):
        if i<train_num:
            shutil.copytree(total_data[i],os.path.join(tartget_dir,'train',os.path.split(total_data[i])[1]))
        elif i<val_num:
            shutil.copytree(total_data[i],os.path.join(tartget_dir,'val',os.path.split(total_data[i])[1]))
        else:
            shutil.copytree(total_data[i],os.path.join(tartget_dir,'test',os.path.split(total_data[i])[1]))