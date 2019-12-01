#!/usr/local/bin/python3
import os
import torch
import torch.utils.data
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from utils import Data
import numpy as np

from models.resnet import resnet18, ResNet, BasicBlock
import matplotlib.pyplot as plt
import numpy as np
import random
import copy


def show_figures(images, title):
    plt.figure(figsize=(8, 10))
    for idx, val in enumerate(images):
        plt.subplot(len(11) / 5 + 1, 5, idx + 1)
        plt.subplots_adjust(top=0.99, bottom=0.01, hspace=1.5, wspace=0.4)
        plt.xticks([], [])
        plt.yticks([], [])
        im, im_class, im_dist = val
        plt.title("{} {}".format(im_class, im_dist))
        plt.imshow(im, cmap="gray")
    plt.tight_layout()
    plt.savefig(title)


transform_test = [transforms.ToTensor()]

transform_train = [
    transforms.RandomCrop(64, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
]


# model = ResNet(BasicBlock, [3, 4, 23, 3], num_classes=1000)
# model._name = "ResNet101"  # resnet18(pretrained=True)
# model.fc = nn.Linear(model.fc.in_features, 200)
model = resnet18(pretrained=False)
model.fc = nn.Linear(2048, 1024) #2048

# Hyperparamters
batch_size = 32
no_epoch = 70
LR = 0.001
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-5)
criterion = nn.TripletMarginLoss(
    margin=1.0
)  # Only change the params, do not change the criterion.

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)
upsample = None  # nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

data = Data(
    batch_size,
    criterion,
    "../data/tiny-imagenet-200/",
    upsample=upsample,
    scheduler=scheduler,
    transform_train=transform_train,
    transform_test=transform_test,
)


start_epoch = 27  # Change me!
should_test = True

if should_test:

    train_embeddings = np.load(
        "trainEmbeddings{}_{}.npy".format(model.name, start_epoch)
    )
    test_embeddings = np.load("testEmbeddings{}_{}.npy".format(model.name, start_epoch))
    train_labels = np.load("embeddingClasses{}_{}.npy".format(model.name, start_epoch))
    test_labels = np.load("testclasses{}_{}.npy".format(model.name, start_epoch))
    # print(
    #     data.knn_accuracy(train_embeddings, test_embeddings, train_labels, test_labels)
    # )
    top, bottom = data.get_top_and_bottom(
        train_embeddings, test_embeddings, train_labels, test_labels
    )
    show_figures(top, "top_images.png")
    show_figures(top, "bottom_images.png")
