#!/usr/local/bin/python3
import os
import torch
import torch.utils.data
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from utils import Data
import numpy as np
import argparse

from models.resnet import resnet18, ResNet, BasicBlock, resnet34

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--no_epoch', type=int, default=75)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--margin', type=float, default=1.0)
parser.add_argument('--step_size', type=int, default=13)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--start_epoch', type=int, default=27)
args = parser.parse_args()

transform_test = [transforms.ToTensor()]

transform_train = [
    transforms.RandomCrop(64, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
]


# model = ResNet(BasicBlock, [3,4,23,3], num_classes=1000)
# model._name = "ResNet"#resnet18(pretrained=True)
# model.fc = nn.Linear(model.fc.in_features, 200)
# model = resnet18(pretrained=False)
# model.fc = nn.Linear(2048, 1024) #2048

model = resnet34(pretrained=False)

# Hyperparamters
# batch_size = 32
# no_epoch = 75
# LR = 0.001
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
criterion = nn.TripletMarginLoss(
    margin=args.margin
)  # Only change the params, do not change the criterion.

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
upsample = None  # nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

data = Data(
    args.batch_size,
    criterion,
    "../data/tiny-imagenet-200/",
    upsample=upsample,
    scheduler=scheduler,
    transform_train=transform_train,
    transform_test=transform_test,
)


# start_epoch = 27  # Change me!


if os.path.exists(
    "models/trained_models/temp_{}_{}.pth".format(model.name, args.start_epoch)
):
    print("found model", model.name)
    model.load_state_dict(
        torch.load(
            "models/trained_models/temp_{}_{}.pth".format(model.name, args.start_epoch)
        )
        # data.test(model)
    )
    # data.test(model)
    data.train(args.no_epoch, model, optimizer, start_epoch=args.start_epoch + 1)
else:
    print("No model found for ", model.name)
    data.train(args.no_epoch, model, optimizer)
