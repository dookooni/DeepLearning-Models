import torch 
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

transforms_ = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

# Data Download & Load
train_dataset = datasets.MNIST(root='./implementation/orthogonal_projection/data', 
                               train=True, 
                               transform=transforms_, 
                               download=True)
test_dataset = datasets.MNIST(root='./implementation/orthogonal_projection/data', 
                              train=False, 
                              transform=transforms_, 
                              download=True)     


# DataLoader
train_loader = DataLoader(
                dataset=train_dataset, 
                batch_size=64, 
                shuffle=True, 
                num_workers=4)

test_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=64, 
                shuffle=False, 
                num_workers=4)

image, label = train_dataset[0]
print(f"{image.shape}")
print(f"{label}")