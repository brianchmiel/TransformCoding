import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
from operations import ReLuPCA
# from torch.nn import nn.ReLU as ReLU
from operations import ReLUCorr as ReLU

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'
}


def flatten(x):
    return x.view(x.size(0), -1)


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, args, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = ReLU(inplace=True) #ReLuPCA(args, planes)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = ReLU(inplace=True) #ReLuPCA(args, planes)

        self.downsample = downsample

        self.stride = stride

    def forward(self, x):

        residue = x

        out = self.relu1(self.bn1(self.conv1(x)))

        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            residue = self.downsample(x)

        out += residue
        out = self.relu2(out)
        return out




class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, args, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = ReLU(inplace=True) #ReLuPCA(args, planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = ReLU(inplace=True) #ReLuPCA(args, planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = ReLU(inplace=True) #ReLuPCA(args, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))

        out = self.relu2(self.bn2(self.conv2(out)))

        out =self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)

        return out



class ResNetImagenet(nn.Module):

    def __init__(self, block, layers, args, zero_init_residual=False):
        super(ResNetImagenet, self).__init__()
        num_classes = 1000
        self.name = args.model
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = ReLU(inplace=True) #ReLuPCA(args, planes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(args, block, 64, layers[0])
        self.layer2 = self._make_layer(args, block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(args, block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(args, block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)



    def _make_layer(self, args, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(args, self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(args, self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def loadPreTrained(self):
        self.load_state_dict(model_zoo.load_url(model_urls[self.name]), False)




def ResNet18(args):
    model = ResNetImagenet(BasicBlock, [2, 2, 2, 2], args)
    return model


def ResNet50(args):
    model = ResNetImagenet(Bottleneck, [3, 4, 6, 3], args)
    return model

def ResNet101(args):
    model = ResNetImagenet(Bottleneck, [3, 4, 23, 3], args)
    return model

# ===================================
# ============== CIFAR ==============
# ===================================


class ResNetCifar(nn.Module):

    def __init__(self, depth, args):
        super(ResNetCifar, self).__init__()
        num_classes = args.nClasses
        assert (depth - 2) % 6 == 0, 'Depth should be 6n + 2'
        n = (depth - 2) // 6
        self.name = args.model
        self.dataset = args.dataset
        block = BasicBlock
        self.inplanes = 64
        fmaps = [64, 128, 256]  # CIFAR10

        self.conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = ReLU(inplace=True) # ReLuPCA(args, 64)

        self.layer1 = self._make_layer(args, block, fmaps[0], n, stride=1)
        self.layer2 = self._make_layer(args, block, fmaps[1], n, stride=2)
        self.layer3 = self._make_layer(args, block, fmaps[2], n, stride=2)

        self.avgpool = nn.AvgPool2d(kernel_size=8, stride=1)
        self.flatten = flatten
        self.fc = nn.Linear(fmaps[2] * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, args, block, planes, blocks, stride=1):
        ''' Between layers convolve input to match dimensions -> stride = 2 '''

        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(block(args, self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(args, self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, ):

        x = self.relu(self.bn(self.conv(x)))  # 32x32

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8

        x = self.avgpool(x)  # 1x1
        x = self.flatten(x)  # Flatten
        x = self.fc(x)  # Dense
        return x

    def loadPreTrained(self):
        preTrainedDir = './preTrained/' + self.name  + '/ckpt.t7'
        checkpoint = torch.load(preTrainedDir)
        self.load_state_dict(checkpoint['net'])


def ResNet20(args):
    return ResNetCifar(depth=20, args=args)

def ResNet56(args):
    return ResNetCifar(depth=56, args=args)