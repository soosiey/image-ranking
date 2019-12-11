import torch
import torchvision
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

from models.main_model import GeneralModel


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class BasicBlock(GeneralModel):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self._name = "BasicBlock"
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(GeneralModel):
    def __init__(self, block, layers, num_classes=100, dropout_prob=0.2, inplanes=32):
        self.inplanes = inplanes
        super(ResNet, self).__init__()
        self._name = "ResNet"
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, inplanes, layers[0], stride=1)
        self.layer2 = self._make_layer(block, inplanes * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, inplanes * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, inplanes * 8, layers[3], stride=2)
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(inplanes * 8 * block.expansion * 4, num_classes)
        self.dropout = torch.nn.Dropout2d(p=dropout_prob)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)
        # print(x.shape)
        x = self.dropout(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # x = self.avgpool(x)
        x = F.max_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
}


def resnet18(pretrained=True):
    # model = torchvision.models.resnet.ResNet(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2])
    model = torchvision.models.resnet.ResNet(
        torchvision.models.resnet.BasicBlock, [2, 2, 2, 2]
    )
    if pretrained:
        model.load_state_dict(
            model_zoo.load_url(
                model_urls["resnet18"], model_dir="~/scratch/resnetModel"
            )
        )

    model.name = "ResNet18"
    return model

def resnet50(pretrained=True):
    # model = torchvision.models.resnet.ResNet(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2])
    model = torchvision.models.resnet.ResNet(
        torchvision.models.resnet.Bottleneck, [3, 4, 6, 3]
    )
    if pretrained:
        model.load_state_dict(
            model_zoo.load_url(
                model_urls["resnet50"], model_dir="~/scratch/resnetModel"
            )
        )

    model.name = "ResNet50"
    return model

def resnet101(pretrained=True):
    # model = torchvision.models.resnet.ResNet(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2])
    model = torchvision.models.resnet.ResNet(
        torchvision.models.resnet.Bottleneck, [3, 4, 23, 3]
    )
    if pretrained:
        model.load_state_dict(
            model_zoo.load_url(model_urls["resnet101"], model_dir="~/scratch/HW4_Model")
        )
    #model = nn.Sequential(model, nn.Linear(1000,200))
    model.name = "ResNet101-Torch"
    return model


def resnet34(pretrained=True):
    model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=1000, inplanes=64)
    if pretrained:
        torch_model = torchvision.models.resnet.ResNet(
            torchvision.models.resnet.BasicBlock, [3, 4, 6, 3]
        )
        torch_model.load_state_dict(
            model_zoo.load_url(model_urls["resnet34"], model_dir="~/scratch/HW4_Model")
        )
        model.layer1.load_state_dict(torch_model.layer1.state_dict())
        model.layer2.load_state_dict(torch_model.layer2.state_dict())
        model.layer3.load_state_dict(torch_model.layer3.state_dict())
        model.layer4.load_state_dict(torch_model.layer4.state_dict())
        # model.fc.load_state_dict(torch_model.fc.state_dict())
    model._name = "ResNet34"
    return model
