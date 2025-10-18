import os
import random

import numpy as np
import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from edl_mirage.utils.helpers import (
    ContrastRescaling,
    GaussianFilter,
    PermutationNoise,
    convert_to_rgb,
)

"""
Data processor for Toy data
"""


def toy_data_processing(num_data, seed):
    random.seed(seed)
    centers_location = np.array([[-2, 3], [0, 0], [2, 3]])
    X_train = []
    y_train = []
    for i, center in enumerate(centers_location):
        cluster_data = np.random.normal(loc=center, scale=0.5, size=(num_data // 3, 2))
        labels = np.full(num_data // 3, i)
        X_train.append(cluster_data)
        y_train.append(labels)
    X_train = np.vstack(X_train)
    y_train = np.hstack(y_train)
    return X_train, y_train


def toy_data_processing_ood(num_data, seed, shell_radius=4, shell_width=0.5):
    random.seed(seed)
    centers_location = np.array([[-2, 3], [0, 0], [2, 3]])
    midpoint = np.mean(centers_location, axis=0)

    X_ood = []
    y_ood = []
    angles = np.random.uniform(low=0, high=2 * np.pi, size=num_data)
    radii = np.random.normal(loc=shell_radius, scale=shell_width, size=num_data)

    cluster_data_x = midpoint[0] + radii * np.cos(angles)
    cluster_data_y = midpoint[1] + radii * np.sin(angles)
    X_ood = np.vstack((cluster_data_x, cluster_data_y)).T
    y_ood = np.full(num_data, 3)

    return X_ood, y_ood


class CustomGaussianDataset(Dataset):
    def __init__(self, num_data, seed):
        self.data, self.labels = toy_data_processing(num_data, seed)
        self.num_samples = len(self.data)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(
            self.labels[idx], dtype=torch.long
        )


class CustomGaussianDataset_OOD(Dataset):
    def __init__(self, num_data, seed):
        self.data, self.labels = toy_data_processing_ood(num_data, seed)
        self.num_samples = len(self.data)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(
            self.labels[idx], dtype=torch.long
        )


"""
Data processor for classical methods (in-distribution data)
"""


class DataProcessing:
    def __init__(
        self, dataset_name, root_path="~/data/", batch_size=128, seed=88, val_ratio=0.2
    ):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.seed = seed
        self.val_ratio = val_ratio
        self.root_path = root_path
        self.path = self.root_path + dataset_name

        if self.dataset_name == "CIFAR10" or self.dataset_name == "CIFAR100":
            self.transform_train = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ),
                ]
            )
            self.transform_test = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ),
                ]
            )
        elif self.dataset_name == "MNIST":
            self.transform_train = transforms.Compose(
                [
                    transforms.Resize(32),
                    transforms.Lambda(convert_to_rgb),
                    transforms.ToTensor(),
                    transforms.Normalize((1 / 2, 1 / 2, 1 / 2), (1 / 2, 1 / 2, 1 / 2)),
                ]
            )
            self.transform_test = transforms.Compose(
                [
                    transforms.Resize(32),
                    transforms.Lambda(convert_to_rgb),
                    transforms.ToTensor(),
                    transforms.Normalize((1 / 2, 1 / 2, 1 / 2), (1 / 2, 1 / 2, 1 / 2)),
                ]
            )
        elif self.dataset_name == "Gaussian":
            print("No transform required!")
        else:
            raise NotImplementedError

    def shuffled_indices(self, dataset):
        num_samples = len(dataset)
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        return torch.randperm(num_samples, generator=generator).tolist()

    def get_split_indices(self, split, dataset, validation_size):
        indices = self.shuffled_indices(dataset)
        print("data number training set", len(indices))
        if split == "validation":
            return indices[:validation_size]
        else:
            return indices[validation_size:]

    def subsample(self, dataset, indices, num_data=None):
        if num_data is not None:
            indices = indices[:num_data]
        return data.Subset(dataset, indices)

    def get_dataloader(self, num_data, shuffle=True):
        if self.dataset_name == "CIFAR10":
            dataset_train = datasets.CIFAR10(
                root=self.path,
                train=True,
                download=True,
                transform=self.transform_train,
            )
            testset = datasets.CIFAR10(
                root=self.path,
                train=False,
                download=True,
                transform=self.transform_test,
            )
        elif self.dataset_name == "CIFAR100":
            dataset_train = datasets.CIFAR100(
                root=self.path,
                train=True,
                download=True,
                transform=self.transform_train,
            )
            testset = datasets.CIFAR100(
                root=self.path,
                train=False,
                download=True,
                transform=self.transform_test,
            )
        elif self.dataset_name == "MNIST":
            dataset_train = datasets.MNIST(
                root=self.path,
                train=True,
                download=True,
                transform=self.transform_train,
            )
            testset = datasets.MNIST(
                root=self.path,
                train=False,
                download=True,
                transform=self.transform_test,
            )
        elif self.dataset_name == "Gaussian":
            dataset_train = CustomGaussianDataset(num_data=10000, seed=self.seed)
            testset = CustomGaussianDataset(num_data=2000, seed=self.seed)
        else:
            raise NotImplementedError

        train_index = self.get_split_indices(
            "train", dataset_train, int(self.val_ratio * len(dataset_train))
        )
        val_index = self.get_split_indices(
            "validation", dataset_train, int(self.val_ratio * len(dataset_train))
        )
        trainset = self.subsample(dataset_train, train_index, num_data)
        valset = self.subsample(dataset_train, val_index)

        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=self.batch_size, shuffle=shuffle, num_workers=8
        )
        valloader = torch.utils.data.DataLoader(
            valset, batch_size=self.batch_size, shuffle=False, num_workers=8
        )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=self.batch_size, shuffle=False, num_workers=8
        )

        return trainloader, valloader, testloader


class OOD_Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        target_ood = 0
        return image, target_ood


"""
Data processor for classical methods (OOD data)
"""


class OOD_DataProcessing:
    def __init__(
        self, dataset_name, root_path="~/data/", batch_size=128, seed=88, ood_size=10000
    ):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.seed = seed
        self.ood_size = ood_size
        self.root_path = root_path
        self.path = self.root_path + dataset_name

        if self.dataset_name in ["Omniglot", "FashionMNIST", "KMNIST"]:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(32),
                    transforms.Lambda(convert_to_rgb),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
        elif self.dataset_name in [
            "CIFAR10",
            "CIFAR100",
            "SVHN",
            "TinyImageNet",
            "LSUN",
        ]:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
        elif self.dataset_name in ["LSUN"]:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(size=(32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
        elif self.dataset_name in ["CIFAR10_noise", "CIFAR100_noise"]:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(32),
                    transforms.ToTensor(),
                    PermutationNoise(),
                    GaussianFilter(),
                    ContrastRescaling(),
                ]
            )
        elif self.dataset_name == "MNIST_noise":
            self.transform = transforms.Compose(
                [
                    transforms.Resize(32),
                    transforms.Lambda(convert_to_rgb),
                    transforms.ToTensor(),
                    PermutationNoise(),
                    GaussianFilter(),
                    ContrastRescaling(),
                ]
            )
        elif self.dataset_name == "Gaussian_OOD":
            print("No transform required!")
        else:
            raise NotImplementedError

    def shuffled_indices(self, dataset):
        num_samples = len(dataset)
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        return torch.randperm(num_samples, generator=generator).tolist()

    def get_split_indices(self, dataset, ood_size):
        indices = self.shuffled_indices(dataset)
        print("data number training set", len(indices))
        return indices[:ood_size]

    def get_dataloader(self, shuffle=True):
        if self.dataset_name == "Omniglot":
            testset = datasets.Omniglot(
                root=self.path,
                background=False,
                download=True,
                transform=self.transform,
            )
        elif self.dataset_name == "KMNIST":
            testset = datasets.KMNIST(
                root=self.path, train=False, download=True, transform=self.transform
            )
        elif self.dataset_name == "SVHN":
            testset = datasets.SVHN(
                root=self.path, split="test", download=True, transform=self.transform
            )
        elif self.dataset_name == "CIFAR10":
            testset = datasets.CIFAR10(
                root=self.path, train=False, download=True, transform=self.transform
            )
        elif self.dataset_name == "CIFAR100":
            testset = datasets.CIFAR100(
                root=self.path, train=False, download=True, transform=self.transform
            )
        elif self.dataset_name == "FashionMNIST":
            testset = datasets.FashionMNIST(
                root=self.path, train=False, download=True, transform=self.transform
            )
        elif self.dataset_name == "TinyImageNet":
            testset = datasets.ImageFolder(
                os.path.join(self.path, "test"), transform=self.transform
            )
        elif self.dataset_name == "LSUN":
            testset = datasets.LSUN(
                self.path, classes=["classroom_val"], transform=self.transform
            )
        elif self.dataset_name == "MNIST_noise":
            testset = datasets.MNIST(
                root=self.root_path + "MNIST",
                train=False,
                download=False,
                transform=self.transform,
            )
        elif self.dataset_name == "CIFAR10_noise":
            testset = datasets.CIFAR10(
                root=self.root_path + "CIFAR10",
                train=False,
                download=False,
                transform=self.transform,
            )
        elif self.dataset_name == "CIFAR100_noise":
            testset = datasets.CIFAR100(
                root=self.root_path + "CIFAR100",
                train=False,
                download=False,
                transform=self.transform,
            )
        elif self.dataset_name == "Gaussian_OOD":
            testset = CustomGaussianDataset_OOD(num_data=2000, seed=self.seed)
        else:
            raise NotImplementedError

        testset = OOD_Dataset(testset)
        ood_index = self.get_split_indices(testset, self.ood_size)
        oodset = data.Subset(testset, ood_index)
        oodloader = torch.utils.data.DataLoader(
            oodset, batch_size=self.batch_size, shuffle=shuffle, num_workers=8
        )

        return oodloader


class EnsembleDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, logits):
        super().__init__()
        self.dataset = dataset
        self.logits = logits

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, target = self.dataset[idx]
        logits = self.logits[idx]
        return image, target, logits


"""
Data processor for distillation based methods (in-distribution data)
"""


class Distill_DataProcessing(DataProcessing):
    def __init__(
        self, dataset_name, root_path="~/data/", batch_size=128, seed=88, val_ratio=0.2
    ):
        super().__init__(dataset_name, root_path, batch_size, seed, val_ratio)

    def get_dataloader(self, num_data, logits_train, logits_val, logits_test):
        if self.dataset_name == "CIFAR10":
            dataset_train = datasets.CIFAR10(
                root=self.path,
                train=True,
                download=True,
                transform=self.transform_train,
            )
            testset = datasets.CIFAR10(
                root=self.path,
                train=False,
                download=True,
                transform=self.transform_test,
            )
        elif self.dataset_name == "CIFAR100":
            dataset_train = datasets.CIFAR100(
                root=self.path,
                train=True,
                download=True,
                transform=self.transform_train,
            )
            testset = datasets.CIFAR100(
                root=self.path,
                train=False,
                download=True,
                transform=self.transform_test,
            )
        elif self.dataset_name == "MNIST":
            dataset_train = datasets.MNIST(
                root=self.path,
                train=True,
                download=True,
                transform=self.transform_train,
            )
            testset = datasets.MNIST(
                root=self.path,
                train=False,
                download=True,
                transform=self.transform_test,
            )
        elif self.dataset_name == "Gaussian":
            dataset_train = CustomGaussianDataset(num_data=10000, seed=self.seed)
            testset = CustomGaussianDataset(num_data=2000, seed=self.seed)
        else:
            raise NotImplementedError

        train_index = self.get_split_indices(
            "train", dataset_train, int(self.val_ratio * len(dataset_train))
        )
        val_index = self.get_split_indices(
            "validation", dataset_train, int(self.val_ratio * len(dataset_train))
        )
        trainset = self.subsample(dataset_train, train_index, num_data)
        valset = self.subsample(dataset_train, val_index)

        trainset = EnsembleDataset(trainset, logits_train)
        valset = EnsembleDataset(valset, logits_val)
        testset = EnsembleDataset(testset, logits_test)

        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=self.batch_size, shuffle=True, num_workers=8
        )
        valloader = torch.utils.data.DataLoader(
            valset, batch_size=self.batch_size, shuffle=False, num_workers=8
        )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=self.batch_size, shuffle=False, num_workers=8
        )

        return trainloader, valloader, testloader


"""
Data processor for distillation based methods (OOD data)
"""


class Distill_OOD_DataProcessing:
    def __init__(
        self, dataset_name, root_path="~/data/", batch_size=128, seed=88, ood_size=10000
    ):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.seed = seed
        self.ood_size = ood_size
        self.root_path = root_path
        self.path = self.root_path + dataset_name

        if self.dataset_name in ["Omniglot", "FashionMNIST", "KMNIST"]:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(32),
                    transforms.Lambda(convert_to_rgb),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
        elif self.dataset_name in [
            "CIFAR10",
            "CIFAR100",
            "SVHN",
            "TinyImageNet",
            "LSUN",
        ]:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
        elif self.dataset_name in ["LSUN"]:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(size=(32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
        elif self.dataset_name in ["CIFAR10_noise", "CIFAR100_noise"]:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(32),
                    transforms.ToTensor(),
                    PermutationNoise(),
                    GaussianFilter(),
                    ContrastRescaling(),
                ]
            )
        elif self.dataset_name == "MNIST_noise":
            self.transform = transforms.Compose(
                [
                    transforms.Resize(32),
                    transforms.Lambda(convert_to_rgb),
                    transforms.ToTensor(),
                    PermutationNoise(),
                    GaussianFilter(),
                    ContrastRescaling(),
                ]
            )
        else:
            raise NotImplementedError

    def shuffled_indices(self, dataset):
        num_samples = len(dataset)
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        return torch.randperm(num_samples, generator=generator).tolist()

    def get_split_indices(self, dataset, ood_size):
        indices = self.shuffled_indices(dataset)
        print("data number training set", len(indices))
        return indices[:ood_size]

    def get_dataloader(self, logits_test):
        if self.dataset_name == "Omniglot":
            testset = datasets.Omniglot(
                root=self.path,
                background=False,
                download=True,
                transform=self.transform,
            )
        elif self.dataset_name == "KMNIST":
            testset = datasets.KMNIST(
                root=self.path, train=False, download=True, transform=self.transform
            )
        elif self.dataset_name == "SVHN":
            testset = datasets.SVHN(
                root=self.path, split="test", download=True, transform=self.transform
            )
        elif self.dataset_name == "CIFAR10":
            testset = datasets.CIFAR10(
                root=self.path, train=False, download=True, transform=self.transform
            )
        elif self.dataset_name == "CIFAR100":
            testset = datasets.CIFAR100(
                root=self.path, train=False, download=True, transform=self.transform
            )
        elif self.dataset_name == "FashionMNIST":
            testset = datasets.FashionMNIST(
                root=self.path, train=False, download=True, transform=self.transform
            )
        elif self.dataset_name == "TinyImageNet":
            testset = datasets.ImageFolder(
                os.path.join(self.path, "test"), transform=self.transform
            )
        elif self.dataset_name == "LSUN":
            testset = datasets.LSUN(
                self.path, classes=["classroom_val"], transform=self.transform
            )
        elif self.dataset_name == "MNIST_noise":
            testset = datasets.MNIST(
                root=self.root_path + "MNIST",
                train=False,
                download=False,
                transform=self.transform,
            )
        elif self.dataset_name == "CIFAR10_noise":
            testset = datasets.CIFAR10(
                root=self.root_path + "CIFAR10",
                train=False,
                download=False,
                transform=self.transform,
            )
        elif self.dataset_name == "CIFAR100_noise":
            testset = datasets.CIFAR100(
                root=self.root_path + "CIFAR100",
                train=False,
                download=False,
                transform=self.transform,
            )
        else:
            raise NotImplementedError

        testset = OOD_Dataset(testset)
        ood_index = self.get_split_indices(testset, self.ood_size)
        oodset = data.Subset(testset, ood_index)
        oodset = EnsembleDataset(oodset, logits_test)
        oodloader = torch.utils.data.DataLoader(
            oodset, batch_size=self.batch_size, shuffle=False, num_workers=8
        )

        return oodloader
