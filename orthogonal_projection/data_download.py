# Download for MNIST dataset

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import sys
import os

transforms_ = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

if __name__ == "__main__":
    if not os.path.exists("./implementation/orthogonal_projection/data"):
        train_dataset = datasets.MNIST(root='./implementation/orthogonal_projection/data', 
                                train=True, 
                                transform=transforms_, 
                                download=True)
        test_dataset = datasets.MNIST(root='./implementation/orthogonal_projection/data', 
                                train=False, 
                                transform=transforms_, 
                                download=True)
