import numpy as np
import torch
import PIL
import argparse
import json
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms, models
from math import ceil


def load_checkpoint(checkpoint_path):
    
    checkpoint = torch.load(checkpoint_path)

    model = models.vgg16(pretrained=True)
    for param in model.parameters(): param.requires_grad = False
    
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    
    
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    
    scheduler = scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=4,gamma=0.1,last_epoch=-1)
    return model, optimizer, scheduler


