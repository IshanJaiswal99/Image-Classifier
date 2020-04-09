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



def save_checkpoint(model,training_image_datasets,option):
    checkpoint = {
    'state_dict': model.state_dict(),
    'model': model,
    'classifier': model.classifier,
    'class_to_idx': training_image_datasets.class_to_idx
    }

    torch.save(checkpoint, option['save_dir'] + 'checkpoint.pth')
    
    
    
    

def build_train_model(option, dataset_sizes, loader, data_sets):
    
    #model = torchvision.models.vgg16(pretrained=True)
    model = eval("torchvision.models.{}(pretrained=True)".format(option['arch']))
    #Freezing parameters
    for params in model.parameters():
        params.requires_grad = False
    #Define Classifier
    model.classifier = nn.Sequential(collections.OrderedDict([
        ('fc1', nn.Linear(25088, 4096, bias=True)),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(p=0.5)),
        ('fc2', nn.Linear(4096, option['hidden_units'], bias=True)),
        ('relu2', nn.ReLU()),
        ('dropout1', nn.Dropout(p=0.25)),
        ('fc3', nn.Linear(option['hidden_units'], 102, bias=True)),
        ('output', nn.LogSoftmax(dim=1))
        ]))

    if option['cuda']:
        device =   torch.device("cuda:0")
    else:
        device =  torch.device("cpu")
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    #Loss and Optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr= option['learning_rate'] )

    since = time.time()
    scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=4,gamma=0.1,last_epoch=-1)
    epochs = option['epochs']
    print_every = 5
    steps = 0
    
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch + 1, epochs))
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train()  
            else:
                model.eval()
            running_loss = 0
            running_corrects = 0
            
            for inputs, labels in loader[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                #Zero Parameter Gradients
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs,labels)
                    
                    if phase == 'train':
                        loss.backward() 
                        optimizer.step()
                    running_loss += loss.item()*inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print('{} loss: {:.4f} Accu: {:.4f}'.format(phase,epoch_loss,epoch_acc))
            best_acc = epoch_acc
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    save_checkpoint(model,data_sets['train'],option)
    
 
