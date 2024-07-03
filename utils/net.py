import math
import timm
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, channels, width, height = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=2)
        proj_value = self.value_conv(x).view(batch_size, -1, width * height)
        out = torch.bmm(attention, proj_value).view(batch_size, channels, width, height)
        out = self.gamma * out + x
        return out

class BinaryClassificationMobileNetV3Large(nn.Module):
    def __init__(self,out_size):
        super(BinaryClassificationMobileNetV3Large, self,).__init__()

        # 加载预训练的MobileNetV3 Large模型
        mobilenet = models.mobilenet_v3_large(pretrained=True)

        # 获取MobileNetV3的特征提取部分（骨干网络）
        self.features = mobilenet.features

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(960, 256),  # MobileNetV3 Large最后一层特征的维度为960
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, out_size),
            nn.Softmax()  # 添加 softmax 操作
        )

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])  # 全局平均池化
        x = self.classifier(x)
        return x

class CustomResNet(nn.Module):
    def __init__(self, num_classes=2,hidden_size=256):
        super(CustomResNet, self).__init__()
        self.num_classes = num_classes
        # 加载预训练的ResNet-50模型
        self.resnet_model = models.resnet50(pretrained=True)
        self.num_features = self.resnet_model.fc.in_features
        self.hidden_size = hidden_size
        
        # 获取ResNet-50的特征提取部分（骨干网络）
        self.backbone = nn.Sequential(*list(self.resnet_model.children())[:-2])
        
        # 自定义前馈神经网络
        self.custom_network = nn.Sequential(
            nn.Linear(self.num_features, self.hidden_size),  # 假设ResNet-50的输出特征维度为2048
            nn.BatchNorm1d(self.hidden_size),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(self.hidden_size, self.num_classes),  # num_classes是你的任务的类别数量
            nn.Softmax() 
        )
    
    def forward(self, x):
        # 使用ResNet-50的特征提取部分
        features = self.backbone(x)
        
        # 全局平均池化
        pooled_features = nn.functional.adaptive_avg_pool2d(features, 1).squeeze(-1).squeeze(-1)
        
        # 将全局平均池化后的特征传递给自定义前馈神经网络
        output = self.custom_network(pooled_features)
        return output

class CustomEfficientNet(nn.Module):
    def __init__(self, num_classes=2, hidden_size=256):
        
        super(CustomEfficientNet, self).__init__()
        # Load pre-trained EfficientNet-B0 model
        self.effnet = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
        self.num_features = self.effnet.num_features
        self.hidden_size = hidden_size
        
        # Custom classifier for binary classification
        self.classifier = nn.Sequential(
            nn.Linear(self.num_features, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, num_classes)
        )
    def forward(self, x):
        # Extract features using EfficientNet
        features = self.effnet(x)
        # Pass the features through the classifier
        output = self.classifier(features)
        return output






class CustomEfficientNetV2(nn.Module):
    def __init__(self, num_classes=2, hidden_size=256):
        super(CustomEfficientNetV2, self).__init__()
        # Load pre-trained EfficientNetV2-Small model
        self.effnetv2 = timm.create_model('tf_efficientnetv2_s', pretrained=True, num_classes=0)  # 使用 EfficientNetV2
        
        self.num_features = self.effnetv2.num_features
        self.hidden_size = hidden_size
        
        # Custom classifier for binary classification
        self.classifier = nn.Sequential(
            nn.Linear(self.num_features, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(self.hidden_size, num_classes),
            nn.Softmax() 
        )

    def forward(self, x):
        # Extract features using EfficientNetV2
        features = self.effnetv2(x)
        
        # Pass the features through the classifier
        output = self.classifier(features)
        
        return output



class CustomEfficientNet_b1(nn.Module):
    def __init__(self, num_classes=2, hidden_size=256):
        super(CustomEfficientNet_b1, self).__init__()
        #加载Effcientnet预训练模型
        self.effnet = timm.create_model('efficientnet_b1', pretrained=True, num_classes=0)  
        self.num_features = self.effnet.num_features
        #分类器
        self.classifier = nn.Sequential(
            nn.Linear(self.num_features, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        # EffcientNet 特征提取
        features = self.effnet(x)
        
        # 分类器
        output = self.classifier(features)
        return output
    
class CustomEfficientNet_b5(nn.Module):
    def __init__(self, num_classes=2, hidden_size=256):
        super(CustomEfficientNet_b5, self).__init__()
        
        # Load pre-trained EfficientNet-B0 model
        self.effnet = timm.create_model('efficientnet_b5', pretrained=True, num_classes=0)  
        
  
        self.num_features = self.effnet.num_features
        
        # Custom classifier for binary classification
        self.classifier = nn.Sequential(
            nn.Linear(self.num_features, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        # Extract features using EfficientNet
        features = self.effnet(x)
        
        # Pass the features through the classifier
        output = self.classifier(features)
        
        return output
    


import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self,classes=5,hidden_size=256,img_size=96):

        # super(Net, self).__init__()
        super(CNN, self).__init__()
        self.img_size = img_size
        # 第一层卷积：输入通道数为3，输出通道数为16，卷积核大小为3x3
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        # 第二层卷积：输入通道数为16，输出通道数为32，卷积核大小为3x3
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        # 第三层卷积：输入通道数为32，输出通道数为64，卷积核大小为3x3
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # 全连接层：输入尺寸根据图像尺寸和卷积层输出决定，假设输入图像尺寸为32x32
        wh=int(img_size/8)
        self.fc1 = nn.Linear(64 * wh*wh, hidden_size)
        self.fc2 = nn.Linear(hidden_size, classes)  # 假设有10个类别

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)  # 展平特征图
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
class CNN_2(nn.Module):
    def __init__(self,classes=5,hidden_size=256,img_size=96):
        super(CNN_2, self).__init__()   # 继承__init__功能
        ## 第一层卷积
        self.conv1 = nn.Sequential(
            # 输入[1,28,28]
            nn.Conv2d(
                in_channels=3,    # 输入图片的高度
                out_channels=16,  # 输出图片的高度
                kernel_size=5,    # 5x5的卷积核，相当于过滤器
                stride=1,         # 卷积核在图上滑动，每隔一个扫一次
                padding=2,        # 给图外边补上0
            ),
            # 经过卷积层 输出[16,28,28] 传入池化层
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)   # 经过池化 输出[16,14,14] 传入下一个卷积
        )
        ## 第二层卷积
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,    # 同上
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            # 经过卷积 输出[32, 14, 14] 传入池化层
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # 经过池化 输出[32,7,7] 传入输出层
        )
        ## 输出层
        # self.output = nn.Linear(in_features=32*int(img_size//4)*int(img_size//4), out_features=classes)
        self.output = nn.Linear(in_features=123008, out_features=classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)           # [batch, 32,7,7]
        x = x.view(x.size(0), -1)   # 保留batch, 将后面的乘到一起 [batch, 32*7*7]
        output = self.output(x)     # 输出[50,10]
        return output
