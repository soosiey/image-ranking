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


transform_test = [transforms.ToTensor()]

transform_train = [
    transforms.RandomCrop(64, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
]

#model = ResNet(BasicBlock, [2,4,4,2], num_classes=200)#resnet18(pretrained=True)
#model.fc = nn.Linear(model.fc.in_features, 200)
model = resnet18(pretrained=True)
model.fc = nn.Linear(2048, 1024) #2048

# Hyperparamters
batch_size = 16
no_epoch = 70
LR = 0.001
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
criterion = nn.TripletMarginLoss(
    margin=1.0
)  # Only change the params, do not change the criterion.
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True) 

data = Data(
    batch_size,
    criterion,
    "../data/tiny-imagenet-200/",
    upsample=upsample,
    scheduler=scheduler,
    transform_train=transform_train,
    transform_test=transform_test,
)


start_epoch = 0  # Change me!
should_train = True
should_test = False
if should_train:
  if os.path.exists("models/trained_models/temp_{}_{}.pth".format(model.name, start_epoch)):
    print("found model")
    model.load_state_dict(
        torch.load(
            "models/trained_models/temp_{}_{}.pth".format(model.name, start_epoch)
        )
    )
    #data.test(model)
    data.train(no_epoch, model, optimizer, start_epoch=start_epoch+1)
  else:
    data.train(no_epoch, model, optimizer)

if should_test:
    if not os.path.exists('testEmbeddings{}_{}.npy'.format(model.name, start_epoch)):
        model.load_state_dict(
            torch.load(
                "models/trained_models/temp_{}_{}.pth".format(model.name, start_epoch)
            )
        )

        data.test(model, epoch)
    train_embeddings = np.load('trainEmbeddings{}_{}.npy'.format(model.name, start_epoch))
    test_embeddings = np.load('testEmbeddings{}_{}.npy'.format(model.name, start_epoch))
    train_labels = np.load('embeddingClasses{}_{}.npy'.format(model.name, start_epoch))
    test_labels = np.load('testclasses{}_{}.npy'.format(model.name, start_epoch))
    print(data.knn_accuracy(train_embeddings, test_embeddings, train_labels, test_labels))

