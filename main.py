#!/usr/local/bin/python3
import os
import torch
import torch.utils.data
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from utils import Data

from models.main_model import GeneralModel as Model

batch_size = 100

transform_test = [transforms.ToTensor()]

transform_train = [
    transforms.RandomCrop(64, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
]


model = Model()

start_epoch = 0
no_epoch = 200
LR = 0.01
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
criterion = nn.CrossEntropyLoss()

data = Data(
    batch_size,
    criterion,
    transform_train=transform_train,
    transform_test=transform_test,
)


if os.path("models/trained_models/temp_{}_{}.model".format(model.name, start_epoch)):
    model = torch.load(
        "models/trained_models/temp_{}_{}.model".format(model.name, start_epoch)
    )
    data.train(no_epoch, model, optimizer, start_epoch=start_epoch)
else:
    data.train(no_epoch, model, optimizer)
