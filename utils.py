#!/usr/local/bin/python3
import os
import torch
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import time
from PIL import Image
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


def pil_loader(path):
    with open(path, "rb") as f:
        IMG = Image.open(f)
        return IMG.convert("RGB")


class TinyImage(Dataset):
    def __init__(
        self,
        root_dir,
        *args,
        train=True,
        num_classes=200,
        transform=None,
        loader=pil_loader,
        **kwargs
    ):
        super(Dataset).__init__(*args, **kwargs)
        self.train = train
        self.root_dir = root_dir
        self.path_to_img = {}
        self.class_to_path = {}
        self.all_images = []
        self.num_classes = num_classes
        self.loader = loader
        self.transform = transform
        for idx, val in enumerate(os.listdir(self.root_dir)):
            curr_dir = os.path.join(self.root_dir, val)
            if "train" in self.root_dir:
                curr_dir = os.path.join(self.root_dir, val, "images")
            curr_images = list(
                map(lambda x: os.path.join(curr_dir, x), os.listdir(curr_dir))
            )
            self.path_to_img[curr_dir] = curr_images
            self.class_to_path[idx] = curr_dir
            self.all_images += list(
                map(lambda x: (x, idx), curr_images)
            )


    def __getitem__(self, idx):
        imp, im_class = self.all_images[idx]
        images = [imp]
        if self.train:
            imp1, im_class = self.all_images[idx]
            imp2 = np.random.choice(self.path_to_img[self.class_to_path[im_class]])
            diff_class = np.random.choice(
                range(0, im_class) + range(im_class + 1, self.num_classes)
            )
            imp3 = np.random.choice(self.path_to_img[self.class_to_path[diff_class]])

            images = [imp1, imp2, imp3]


        for idx, imp in enumerate(images):
            images[idx] = self.loader(imp)
            if self.transform:
                images[idx] = self.transform(images[idx])
        return images, im_class

    def __len__(self):
        return len(self.all_images)


class Data:
    def __init__(
        self,
        batch_size,
        criterion,
        data_dir,
        upsample=None,
        scheduler=None,
        transform_train=None,
        transform_test=None,
    ):
        self.batch_size = batch_size
        self.criterion = criterion
        self.scheduler = scheduler
        if upsample is not None:
            self.upsample = upsample
        else:
            self.upsample = (lambda x: x)

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

        train_dir = os.path.join(data_dir, "train/")
        train_dataset = TinyImage(train_dir, transform=transform_train)

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=8
        )

        train_dataset = TinyImage(train_dir, transform=transform_train, train=False)

        self.emb_train = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=8
        )

        val_dir = os.path.join(data_dir, "val/")
        if "val_" in os.listdir(val_dir + "images/")[0]:
            create_val_folder(val_dir)
        val_dir += "images"

        val_dataset = TinyImage(val_dir, transform=transform_test, train=False)
        self.val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=8
        )

    def train(self, no_epoch, model, optimizer, start_epoch=0, should_save=True):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        test_acc_list = []
        train_acc_list = []
        time_list = []
        start_time = time.time()
        losses = []
        for epoch in range(start_epoch, no_epoch):
            training_p = []
            classes_p = []
            model.train()
            train_accu = []
            epochLosses = []
            for batch_idx, ((im1, im2, im3), c) in enumerate(self.train_loader):
                im1, im2, im3 = (
                    self.upsample(Variable(im1).to(device)),
                    self.upsample(Variable(im2).to(device)),
                    self.upsample(Variable(im3).to(device)),
                )
                print("Running P model", epoch, batch_idx, end="\r")
                P = model(im1)
                training_p += list(P.data.cpu().numpy())
                classes_p += c
                print("Running Q model", epoch, batch_idx, end="\r")
                Q = model(im2)
                print("Running R model", epoch, batch_idx, end="\r")
                R = model(im3)
                loss = self.criterion(P, Q, R)
                optimizer.zero_grad()
                loss.backward()

                optimizer.step()
                epochLosses.append(loss.item())
            l = np.mean(epochLosses)
            print('Loss for epoch' + epoch + ': ' + l)
            losses.append(l)
            np.save(training_p, 'trainEmbeddings{}_{}'.format(model.name, epoch))
            np.save(classes_p, 'embeddingClasses{}_{}'.format(model.name, epoch))

                # prediction = output.data.max(1)[1]
                # accuracy = (
                #     float(prediction.eq(Y_train_batch.data).sum())
                #     / float(self.batch_size)
                # ) * 100.0
                # train_accu.append(accuracy)
            if self.scheduler is not None:
                self.scheduler.step()
            # train_acc = np.mean(train_accu)

            # with torch.no_grad():
            #     model.eval()
            #     test_accu = []
            #     for batch_idx, (X_test_batch, Y_test_batch) in enumerate(
            #         self.val_loader
            #     ):
            #         X_test_batch, Y_test_batch = (
            #             Variable(X_test_batch).to(device),
            #             Variable(Y_test_batch).to(device),
            #         )
            #         output = model(X_test_batch)
            #         prediction = output.data.max(1)[1]
            #         accuracy = (
            #             float(prediction.eq(Y_test_batch.data).sum())
            #             / float(self.batch_size)
            #         ) * 100.0
            #         test_accu = []
            #     test_acc = np.mean(test_accu)
            # train_acc_list.append(train_acc)
            # test_acc_list.append(test_acc)
            # time_list.append(time.time() - start_time)

            if (epoch + 1) % 1 == 0 and should_save:
                print("Saving Model")
                torch.save(
                    model.state_dict(),
                    "models/trained_models/temp_{}_{}.pth".format(model.name, epoch),
                )
                torch.save(
                    optimizer,
                    "models/trained_models/temp_{}_{}.state".format(model.name, epoch),
                )
                # data = [train_acc_list, test_acc_list]
                # data = np.asarray(data)
                # np.save(
                #     "models/trained_models/temp_{}_{}.npy".format(model.name, epoch),
                #     data,
                # )

        if should_save:
            torch.save(
                model.state_dict(), "models/trained_models/{}.pth".format(model.name)
            )
            np.save(losses,'Losses{}'.format(model.name))
            # np.save("models/trained_models/{}_{}.npy".format(model.name, epoch), data)

    def test(self, model):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        loss_arr = []
        for batch_idx, ((im1, im2, im3), j) in enumerate(self.train_loader):
            im1, im2, im3 = (
                self.upsample(Variable(im1).to(device)),
                self.upsample(Variable(im2).to(device)),
                self.upsample(Variable(im3).to(device)),
            )
            P = model(im1)
            Q = model(im2)
            R = model(im3)
            loss = self.criterion(P, Q, R)
            loss_arr.append(loss.item())
        print(np.mean(loss_arr))

    def train_emb(self, model):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        for batch_idx, (im, j) in enumerate(self.emb_train):
            im = (self.upsampl(Varible(im).to(device)))
            val = model(im)
            # TODO: Naveen

