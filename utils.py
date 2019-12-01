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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


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
            self.all_images += list(map(lambda x: (x, idx), curr_images))

    def __getitem__(self, idx):
        imp, im_class = self.all_images[idx]
        images = [imp]
        if self.train:
            imp1, im_class = self.all_images[idx]
            imp2 = np.random.choice(self.path_to_img[self.class_to_path[im_class]])
            diff_class = np.random.choice(
                list(range(0, im_class)) + list(range(im_class + 1, self.num_classes))
            )
            imp3 = np.random.choice(self.path_to_img[self.class_to_path[diff_class]])

            images = [imp1, imp2, imp3]
        else:
            imp = self.loader(imp)
            if self.transform:
                imp = self.transform(imp)
            return imp, im_class

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
            self.upsample = lambda x: x

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
        print("Training started")
        for epoch in range(start_epoch, no_epoch):
            model.train()
            train_accu = []
            epochLosses = []
            for batch_idx, ((im1, im2, im3), c) in enumerate(self.train_loader):
                im1, im2, im3 = (
                    self.upsample(Variable(im1).to(device)),
                    self.upsample(Variable(im2).to(device)),
                    self.upsample(Variable(im3).to(device)),
                )
                Q = model(im1)
                P = model(im2)
                N = model(im3)
                loss = self.criterion(Q, P, N)
                optimizer.zero_grad()
                loss.backward()

                optimizer.step()
                epochLosses.append(loss.item())
            mean_loss = np.mean(epochLosses)
            print("Loss for epoch", epoch, ":", mean_loss)
            losses.append(mean_loss)

            if self.scheduler is not None:
                self.scheduler.step()

            if (epoch + 1) % 1 == 0 and should_save:
                print("Saving Model")
                torch.save(
                    model.state_dict(),
                    "models/trained_models/temp_{}_{}.pth".format(model.name, epoch),
                )

                # torch.save(
                #    optimizer,
                #    "models/trained_models/temp_{}_{}.state".format(model.name, epoch),
                # )

        if should_save:
            torch.save(
                model.state_dict(), "models/trained_models/{}.pth".format(model.name)
            )

            np.save("losses_{}.npy".format(model.name), losses)

    def test(self, model, epoch):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        testing_q = []
        classes_q = []
        print("Creating test embeddings")
        for batch_idx, (im1, j) in enumerate(self.val_loader):
            im1 = self.upsample(Variable(im1).to(device))
            Q = model(im1)
            testing_q += list(Q.data.cpu().numpy())
            classes_q += list(j.data.cpu().numpy())
        np.save("embeddings/test_{}_{}.npy".format(model.name, epoch), testing_q)
        np.save("embeddings/test_labels_{}_{}.npy".format(model.name, epoch), classes_q)

    def train_emb(self, model, epoch):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        testing_q = []
        classes_q = []
        print("Creating Train Embeddings")
        for batch_idx, ((im, _, _), c) in enumerate(self.train_loader):
            im = self.upsample(Variable(im).to(device))
            val = model(im)
            testing_q += list(val.data.cpu().numpy())
            classes_q += list(c.data.cpu().numpy())
            if batch_idx % 13 == 0:
                print("Finishing with", batch_idx)
        np.save("embeddings/train_{}_{}.npy".format(model.name, epoch), testing_q)
        np.save(
            "embeddings/train_labels_{}_{}.npy".format(model.name, epoch), classes_q
        )

    def knn_accuracy(
        self, train_embeddings, test_embeddings, train_labels, test_labels, k=30
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_embeddings = torch.from_numpy(train_embeddings).float().to(device)
        train_labels = torch.from_numpy(train_labels).float().to(device)
        test_labels = torch.from_numpy(test_labels).float().to(device)
        accuracies = []
        tc = 0
        for idx, test in enumerate(test_embeddings):
            test = torch.from_numpy(test).float().to(device)
            dist = torch.sum((train_embeddings - test).pow(2), dim=1)
            _, ind = sort_dist = torch.topk(dist, k, largest=False)
            count = torch.sum(train_labels[ind] == test_labels[idx])

            if count.item() > 0:
                tc += 1
            accuracies.append(count.item() / float(k))
        # knn = KNeighborsClassifier(n_neighbors = k, algorithm = 'kd_tree')
        # knn.fit(train_embeddings, train_labels)
        # predicted_labels = knn.predict_proba(test_embeddings)
        # accuracy = accuracy_score(y_true = test_labels, y_pred = predicted_labels)
        # accuracies = []
        # for idx, sample in enumerate(predicted_labels):
        #    acc = sample[test_labels[idx]]
        #    print("acc", acc, "label", test_labels[idx])
        #    accuracies.append(acc)
        return np.mean(accuracies), tc / 10000.0

    def get_top_and_bottom(
        self, train_embeddings, test_embeddings, train_labels, test_labels, k=10
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_embeddings = torch.from_numpy(train_embeddings).float().to(device)
        train_labels = torch.from_numpy(train_labels).float().to(device)
        test_labels = torch.from_numpy(test_labels).float().to(device)
        images = []
        chosen_classes = set()

        while len(images) < 5:
            im_class = np.random.choice(
                set(range(self.val_loader.num_classes)) - chosen_classes
            )
            chosen_classes.add(im_class)
            imp = self.val_loader.path_to_img[self.val_loader.class_to_path[im_class]]
            len_imp = len(imp)
            idx = np.random.choice(range(len_imp))
            imp = imp[idx]
            im = self.val_loader.loader(imp)
            if self.val_loader.transform:
                im = self.val_loader.transform(im)
            images.append((im, im_class, len_imp * im_class + idx))

        top_images = []
        for im, im_class, im_idx in images:
            test = test_embeddings[im_idx]
            test = torch.from_numpy(test).float().to(device)
            dist = torch.sum((train_embeddings - test).pow(2), dim=1)
            _, ind = torch.topk(dist, k, largest=False)
            top_images.append((im, im_class, im_idx))
            for idx in ind:
                im1, im_class1 = self.val_loader.__getitem__(idx)
                top_images.append((im1, im_class1, idx))
                top_images = []

        bottom_images = []
        for im, im_class, im_idx in images:
            test = test_embeddings[im_idx]
            test = torch.from_numpy(test).float().to(device)
            dist = torch.sum((train_embeddings - test).pow(2), dim=1)
            _, ind = torch.topk(dist, k, largest=True)
            bottom_images.append((im, im_class, im_idx))
            for idx in ind:
                im1, im_class1 = self.val_loader.__getitem__(idx)
                bottom_images.append((im1, im_class1, idx))
            # count = torch.sum(train_labels[ind] == test_labels[im_idx])

            # if count.item() > 0:
            # tc += 1
            # accuracies.append(count.item()/float(k))
        # knn = KNeighborsClassifier(n_neighbors = k, algorithm = 'kd_tree')
        # knn.fit(train_embeddings, train_labels)
        # predicted_labels = knn.predict_proba(test_embeddings)
        # accuracy = accuracy_score(y_true = test_labels, y_pred = predicted_labels)
        # accuracies = []
        # for idx, sample in enumerate(predicted_labels):
        #    acc = sample[test_labels[idx]]
        #    print("acc", acc, "label", test_labels[idx])
        #    accuracies.append(acc)
        return top_images, bottom_images
