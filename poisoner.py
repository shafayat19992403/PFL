from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from utils.util_class import CNNModel
from utils.util_function import *
import torch


# Load the original CIFAR-10 dataset
original_trainset = CIFAR10(root='./data', train=True, download=True, transform=ToTensor())

# Poison the dataset
poison_rate = 0.1  # 10% of the data will be poisoned
poisoned_trainset = poison_CIFAR10_dataset(original_trainset, poison_rate=poison_rate)


torch.save(poisoned_trainset, './data/poisoned_cifar10_trainset.pt')