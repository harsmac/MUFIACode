"""
File defining the data loading class for various datasets used in the project.
"""

from torchvision import datasets
import torchvision
import sys
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils
from PIL import Image

# fix torch seed
torch.manual_seed(42)
# fix cuda seed
torch.cuda.manual_seed(42)

torch.cuda.empty_cache()
sys.path.append("../")

import torch
from torch.utils.data import Dataset
import configs


class DataLoading:
    def __init__(self, params):
        self.dataset_name = params["dataset"]
        self.batch_size = params["batch_size"]
        self.shuffle = False  # to keep consistency

    def get_data(self):
        if self.dataset_name == "cifar10":
            return self.cifar10_loading()
        elif self.dataset_name == "imagenet":
            return self.imagenet_loading()
        elif self.dataset_name == "cifar100":
            return self.cifar100_loading()
        else:
            raise ValueError("Dataset not supported")

    def cifar10_loading(self):
        root_path = configs.dataset_paths["cifar10"]
        transform_test = transforms.Compose([transforms.ToTensor(),])
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )

        testset = torchvision.datasets.CIFAR10(
            root=root_path, train=False, download=True, transform=transform_test,
        )

        trainset = torchvision.datasets.CIFAR10(
            root=root_path, train=True, download=True, transform=transform_train,
        )

        testloader = torch.utils.data.DataLoader(
            testset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=2
        )

        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=2
        )

        print("Loading data from {}".format(self.dataset_name.upper()))

        return trainset, trainloader, testset, testloader

    def cifar100_loading(self):
        root_path = configs.dataset_paths["cifar100"]
        transform_test = transforms.Compose([transforms.ToTensor(),])
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )

        testset = torchvision.datasets.CIFAR100(
            root=root_path, train=False, download=True, transform=transform_test,
        )

        trainset = torchvision.datasets.CIFAR100(
            root=root_path, train=True, download=True, transform=transform_train,
        )

        testloader = torch.utils.data.DataLoader(
            testset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=2
        )

        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=2
        )

        print("Loading data from {}".format(self.dataset_name.upper()))

        return trainset, trainloader, testset, testloader

    def imagenet_loading(self):
        # from madry/robustness
        test_transform = transforms.Compose(
            [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),]
        )
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        imagenet_path_val = configs.dataset_paths["imagenet_val"]
        imagenet_path = configs.dataset_paths["imagenet_train"]

        trainset = datasets.ImageFolder(imagenet_path, train_transform,)
        testset = datasets.ImageFolder(imagenet_path_val, test_transform,)

        testloader = torch.utils.data.DataLoader(
            testset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            pin_memory=True,
            num_workers=8,
        )

        trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            pin_memory=True,
            num_workers=8,
        )
        print("Loading data from {}".format(self.dataset_name.upper()))
        return trainset, trainloader, testset, testloader
