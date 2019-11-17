#!/usr/local/bin/python3
import os
import torch
import torch.utils.data
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from utils import Data

from models.resnet import resnet18


transform_test = [transforms.ToTensor()]

transform_train = [
    transforms.RandomCrop(64, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
]

model = resnet18(pretrained=True)

# Hyperparamters
batch_size = 100
no_epoch = 200
LR = 0.01
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
criterion = nn.TripletMarginLoss(
    margin=1.0
)  # Only change the params, do not change the criterion.
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

data = Data(
    batch_size,
    criterion,
    scheduler=scheduler,
    transform_train=transform_train,
    transform_test=transform_test,
)


start_epoch = 0  # Change me!
if os.path("models/trained_models/temp_{}_{}.model".format(model.name, start_epoch)):
    model.load_state_dict(
        torch.load(
            "models/trained_models/temp_{}_{}.pth".format(model.name, start_epoch)
        )
    )
    data.train(no_epoch, model, optimizer, start_epoch=start_epoch)
else:
    data.train(no_epoch, model, optimizer)
