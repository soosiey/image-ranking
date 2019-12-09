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
parser.add_argument('--resnet', type=int, default=18)
parser.add_argument('--num_classes', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--no_epoch', type=int, default=75)
# parser.add_argument('--momentum', type=float, default=0.9)
# parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--margin', type=float, default=1.0)
# parser.add_argument('--step_size', type=int, default=13)
# parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--start_epoch', type=int, default=0)


transform_test = [transforms.ToTensor()]

transform_train = [
    transforms.RandomCrop(64, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
]
# upsample = None
# model = ResNet(BasicBlock, [2,4,4,2], num_classes=200)#resnet18(pretrained=True)
# model.fc = nn.Linear(model.fc.in_features, 200)
# model = resnet18(pretrained=False)
# model.fc = nn.Linear(2048, 1024) #2048
# upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

# model = ResNet(BasicBlock, [3,4,23,3], num_classes=1000)
# model._name = "ResNet101"

if args.resenet == 0:
    model = ResNet(BasicBlock, [2,4,4,2], num_classes=args.num_classes)
    upsample = None
elif args.resenet == 18:
    model = resnet18(pretrained=False)
    upsample = None
elif args.resenet == 34:
    model = resnet34(pretrained=False)
    upsample = nn.Upsample(scale_factor=3.5, mode='bilinear', align_corners=True)
elif args.resenet == 101:
    model = resnet101(pretrained=False)
    upsample = nn.Upsample(scale_factor=3.5, mode='bilinear', align_corners=True)


# Hyperparamters
batch_size = args.batch_size
no_epoch = args.no_epoch
LR = args.lr
criterion = nn.TripletMarginLoss(
    margin=args.margin
)  # Only change the params, do not change the criterion.

data = Data(
    batch_size,
    criterion,
    "../data/tiny-imagenet-200/",
    upsample=upsample,
    transform_train=transform_train,
    transform_test=transform_test,
)


start_epoch = args.start_epoch  # Change me!

model.load_state_dict(
    torch.load("models/trained_models/temp_{}_{}.pth".format(model.name, start_epoch))
)
print("Retrieved model", model.name)
if not os.path.exists("embeddings/test_{}_{}.npy".format(model.name, start_epoch)):
    data.train_emb(model, start_epoch)
    data.test(model, start_epoch)
train_embeddings = np.load("embeddings/train_{}_{}.npy".format(model.name, start_epoch))
test_embeddings = np.load("embeddings/test_{}_{}.npy".format(model.name, start_epoch))
train_labels = np.load(
    "embeddings/train_labels_{}_{}.npy".format(model.name, start_epoch)
)
test_labels = np.load(
    "embeddings/test_labels_{}_{}.npy".format(model.name, start_epoch)
)
print(data.knn_accuracy(train_embeddings, test_embeddings, train_labels, test_labels))
