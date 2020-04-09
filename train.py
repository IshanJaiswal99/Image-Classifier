# Imports everything
import numpy as np
import matplotlib.pyplot as plt
import os, random
import json
import collections
import time
import shutil, argparse
import PIL
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import variable
import torchvision
from torchvision import datasets, transforms, models

def Createtransforms(data_dir):
    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    training_transforms = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456, 0.406], [0.456, 0.406])
    ])

    validation_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456, 0.406], [0.456, 0.406])
    ])

    testing_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456, 0.406], [0.456, 0.406])
    ])

    # TODO: Load the datasets with ImageFolder
    training_image_datasets = datasets.ImageFolder(train_dir, transform=training_transforms) 
    validation_image_datasets = datasets.ImageFolder(valid_dir, transform=validation_transforms) 
    testing_image_datasets = datasets.ImageFolder(test_dir, transform=testing_transforms) 

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainingDataLoader = torch.utils.data.DataLoader(training_image_datasets, batch_size=32, shuffle=True)
    validationDataLoader = torch.utils.data.DataLoader(validation_image_datasets, batch_size=32, shuffle=True)
    testingDataLoader = torch.utils.data.DataLoader(testing_image_datasets, batch_size=32, shuffle=True)
    loader = {'train': trainingDataLoader, 'valid':validationDataLoader, 'test': testingDataLoader }

    dataset_sizes = {'train': len(training_image_datasets), 
                     'valid': len(validation_image_datasets),
                     'test': len(testing_image_datasets)}
    data_sets = {'train': training_image_datasets, 
                     'valid': validation_image_datasets,
                     'test': testing_image_datasets}
    import json

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    return dataset_sizes, loader, data_sets


