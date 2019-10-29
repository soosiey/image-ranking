#!/usr/local/bin/python3
import os
import torch
import torch.utils.data
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import time
import numpy as np


def create_val_folder(val_dir):
    """
    This method is responsible for separating validation images into separate sub folders
    """
    path = os.path.join(val_dir, "images")  # path where validation data is present now
    filename = os.path.join(
        val_dir, "val_annotations.txt"
    )  # file where image2class mapping is present
    fp = open(filename, "r")  # open file in read mode
    data = fp.readlines()  # read line by line

    # Create a dictionary with image names as key and corresponding classes as values
    val_img_dict = {}
    for line in data:
        words = line.split("\t")
        val_img_dict[words[0]] = words[1]
    fp.close()
    # Create folder if not present, and move image into proper folder
    for img, folder in val_img_dict.items():
        newpath = os.path.join(path, folder)
        if not os.path.exists(newpath):  # check if folder exists
            os.makedirs(newpath)
        if os.path.exists(
            os.path.join(path, img)
        ):  # Check if image exists in default directory
            os.rename(os.path.join(path, img), os.path.join(newpath, img))
    return


class Data:
    def __init__(
        self,
        batch_size,
        criterion,
        scheduler=None,
        transform_train=None,
        transform_test=None,
    ):
        self.batch_size = batch_size
        self.criterion = criterion
        self.scheduler = scheduler

        if transform_train is None:
            transform_train = [
                transforms.RandomCrop(64, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        if transform_test is None:
            transform_test = [transforms.ToTensor()]

        transform_train = transforms.Compose(transform_train)

        transform_test = transforms.Compose(transform_test)

        train_dir = "./dataset/tiny-imagenet-200/train/"
        train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=8
        )

        val_dir = "./dataset/tiny-imagenet-200/val/"
        if "val_" in os.listdir(val_dir + "images/")[0]:
            create_val_folder(val_dir)
            val_dir = val_dir + "images/"
        else:
            val_dir = val_dir + "images/"
        val_dataset = datasets.ImageFolder(val_dir, transform=transform_test)
        self.val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=8
        )

    def train(self, model, optimizer, start_epoch=0, should_save=True):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        test_acc_list = []
        train_acc_list = []
        time_list = []
        start_time = time.time()
        for epoch in range(start_epoch, self.no_epoch):
            model.train()
            train_accu = []
            for batch_idx, (X_train_batch, Y_train_batch) in enumerate(
                self.train_loader
            ):
                X_train_batch, Y_train_batch = (
                    Variable(X_train_batch).to(device),
                    Variable(Y_train_batch).to(device),
                )
                output = model(X_train_batch)
                loss = self.criterion(output, Y_train_batch)
                self.optimizer.zero_grad()
                loss.backward()

                self.optimizer.step()
                prediction = output.data.max(1)[1]
                accuracy = (
                    float(prediction.eq(Y_train_batch.data).sum())
                    / float(self.batch_size)
                ) * 100.0
                train_accu.append(accuracy)
            if self.scheduler is not None:
                self.scheduler.step()
            train_acc = np.mean(train_accu)

            with torch.no_grad():
                model.eval()
                test_accu = []
                for batch_idx, (X_test_batch, Y_test_batch) in enumerate(
                    self.val_loader
                ):
                    X_test_batch, Y_test_batch = (
                        Variable(X_test_batch).to(device),
                        Variable(Y_test_batch).to(device),
                    )
                    output = model(X_test_batch)
                    prediction = output.data.max(1)[1]
                    accuracy = (
                        float(prediction.eq(Y_test_batch.data).sum())
                        / float(self.batch_size)
                    ) * 100.0
                    test_accu = []
                test_acc = np.mean(test_accu)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            time_list.append(time.time() - start_time)

            if (epoch + 1) % 15 == 0 and should_save:
                torch.save(
                    model,
                    "models/trained_models/temp_{}_{}.model".format(model.name, epoch),
                )
                torch.save(
                    optimizer,
                    "models/trained_models/temp_{}_{}.state".format(model.name, epoch),
                )
                data = [train_acc_list, test_acc_list]
                data = np.asarray(data)
                np.save(
                    "models/trained_models/temp_{}_{}.npy".format(model.name, epoch),
                    data,
                )
        if should_save:
            torch.save(model, "models/trained_models/{}.model".format(model.name))
            np.save("models/trained_models/{}_{}.npy".format(model.name, epoch), data)
