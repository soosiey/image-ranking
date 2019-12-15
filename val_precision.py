#!/usr/local/bin/python3
import os
import torch
import torch.utils.data
import torchvision.transforms as transforms
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
from utils import Data, pil_loader
import numpy as np

from models.resnet import resnet50
from PIL import Image

def get_image(path):
    im = pil_loader(path)
    im = im.convert("RGB")
    im = transform_train(im)
    im = torch.reshape(im, (1,) + im.shape)
    return im

transform_train = [
    transforms.Resize(size = (64, 64)),
    transforms.ToTensor()
]
transform_train = transforms.Compose(transform_train)
upsample = nn.Upsample(scale_factor=3.5, mode='bilinear', align_corners=True)

model = resnet50(pretrained=False)
model.fc = nn.Linear(2048,2048)

start_epoch = 9  # Change me!

model.load_state_dict(
    torch.load("models/trained_models/temp_{}_{}.pth".format(model.name, start_epoch))
)
model.to("cuda")
upsample = nn.Upsample(scale_factor=3.5, mode='bilinear')
os.chdir("../validation_data/")
model.eval()
val_T_F = []

for i in range(1, 5034):
    im1 = get_image("q_" + str(i) + ".jpg")
    im2 = get_image("p_" + str(i) + ".jpg")
    im3 = get_image("n_" + str(i) + ".jpg")
    im1, im2, im3 = (
        upsample(im1.to("cuda")),
        upsample(im2.to("cuda")),
        upsample(im3.to("cuda")),
    )
    Q = model(im1)
    P = model(im2)
    N = model(im3)
    Q = Q.reshape((2048, ))
    P = P.reshape((2048, ))
    N = N.reshape((2048, ))
    Q_P_sq = torch.sum((Q-P)**2).item()
    Q_N_sq = torch.sum((Q-N)**2).item()
    val_T_F.append(Q_P_sq < Q_N_sq)

print(sum(val_T_F)/len(val_T_F))