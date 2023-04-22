import torch
import torch.nn as nn
from torchvision.models import resnet18
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, nclass=5, feat_dim=128, layers=[2, 2, 2, 2], align="CONV"):
        super(ResNet, self).__init__()
        net = resnet18(pretrained=True)
        self.extrabit = 0
        self.inplanes = 64
        self.align = align
        #   reduce the kernel-size and stride of ResNet on cifar datasets.
        self.conv1 = net.conv1
        self.bn1 = net.bn1
        self.relu = net.relu

        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        self.avgpool = net.avgpool

        self.type = 'ce'  #contrast or ce

        self.head = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, feat_dim)
        )
        self.linear = nn.Linear(512, nclass)


    def set_type(self):
        self.type = 'ce'

    def forward(self, x): #x(2*batch,3,32,32)
        x = self.conv1(x) #(2*batch,64,32,32)
        x = self.bn1(x) #
        x = self.relu(x) #
        x = self.layer1(x) #(2*batch,64,32,32)
        x = self.layer2(x) #(2*batch,128,16,16)
        x = self.layer3(x) #(2*batch,256,8,8)
        x = self.layer4(x) #(2*batch,512,4,4)
        x = self.avgpool(x) #(batch,512,1,1)
        x = x.view(x.size(0),-1) #(batch,512,1,1)->(batch,512)
        if self.type == 'contrast':
            x = F.normalize(self.head(x), dim=1)
        else:
            x = self.linear(x)
        return x


class LinearClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self, name='resnet50', num_classes=10):
        super(LinearClassifier, self).__init__()
        feat_dim = ResNet()
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, features):
        return self.fc(features)