#!/usr/local/bin/python3
import os
import torch
import torch.utils.data
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from utils import Data
import numpy as np

from models.resnet import resnet18, ResNet, BasicBlock, resnet34, resnet50
import numpy as np
import random
import copy

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


def show_figures(images, title):
    # plt.figure(figsize=(8, 10))
    fig = plt.figure(figsize=(20, 20))
    gs = gridspec.GridSpec(5, 11)
    gs.update(wspace=0.5, hspace=1.6, right=0.9)
    for idx, val in enumerate(images):
        # plt.subplot(11 / 5 + 1, 5, idx + 1)
        # plt.subplots_adjust(top=0.99, bottom=0.01, hspace=1.5, wspace=0.4)
        ax = plt.subplot(gs[idx])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect("equal")
        im, im_class, im_dist = val
        ax.title.set_text("{} {}".format(im_class, round(im_dist, 2)))
        plt.imshow(im)
    # plt.tight_layout()
    plt.savefig(title)


transform_test = [transforms.ToTensor()]

transform_train = [
    #transforms.RandomCrop(64, padding=4),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
]

upsample = nn.Upsample(scale_factor=3.5, mode='bilinear')

# model = ResNet(BasicBlock, [3, 4, 23, 3], num_classes=1000)
# model._name = "ResNet101"  # resnet18(pretrained=True)
# model.fc = nn.Linear(model.fc.in_features, 200)
# model = resnet18(pretrained=False)
# model.fc = nn.Linear(2048, 1024)  # 2048
# upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
model = resnet50(pretrained=False)
model.fc = nn.Linear(2048,2048)
# Hyperparamters
batch_size = 10
no_epoch = 70
LR = 0.001
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-5)
criterion = nn.TripletMarginLoss(
    margin=1.0
)  # Only change the params, do not change the criterion.

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

data = Data(
    batch_size,
    criterion,
    "../data/tiny-imagenet-200/",
    upsample=upsample,
    scheduler=scheduler,
    transform_train=transform_train,
    transform_test=transform_test,
)


start_epoch = 9  # Change me!

if not os.path.exists("embeddings/test_{}_{}.npy".format(model.name, start_epoch)):
    print("Please test your model first then graph it!")
    exit()

train_embeddings = np.load("embeddings/train_{}_{}.npy".format(model.name, start_epoch))
test_embeddings = np.load("embeddings/test_{}_{}.npy".format(model.name, start_epoch))
train_labels = np.load(
    "embeddings/train_labels_{}_{}.npy".format(model.name, start_epoch)
)
test_labels = np.load(
    "embeddings/test_labels_{}_{}.npy".format(model.name, start_epoch)
)
print(
    data.knn_accuracy(
        train_embeddings, test_embeddings, train_labels, test_labels, k=10
    )
)
top, bottom = data.get_top_and_bottom(
    train_embeddings, test_embeddings, train_labels, test_labels
)
show_figures(top, "images/top_images.png")
show_figures(bottom, "images/bottom_images.png")
