#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 16:11:07 2020

@author: briardoty
"""
import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import os
import math
import time
import copy
import torch.optim as optim
from torch.optim import lr_scheduler


nets = {
    "vgg16": {
        "layers_of_interest": {
            "conv2": 2,
            "conv8": 17,
            "conv9": 19,
            "conv10": 21,
            "conv11": 24,
            "conv12": 26,
            "conv13": 28
        }
    },
    "alexnet": {
        "layers_of_interest": {
            "conv2": 3
        }
    }
}

def get_net_tag(net_name, snapshot):
    net_tag = f"{net_name}"
    if (snapshot is not None):
        net_tag += f"_epoch_{snapshot}"
        
    return net_tag

def load_test_stimuli(data_dir, img_xy = 200):
    # pull in images
    data_transforms = transforms.Compose([
        transforms.CenterCrop(img_xy),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225])
    ])
    
    image_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, "stimuli/"),
        transform=data_transforms)
    
    image_loader = torch.utils.data.DataLoader(image_dataset)
    
    return (image_dataset, image_loader)

class NetManager():
    
    def __init__(self, net_name, n_classes, data_dir, pretrained=False):
        self.net_name = net_name
        self.pretrained = pretrained
        self.data_dir = os.path.expanduser(data_dir)
        self.n_classes = n_classes
        self.epoch = 0
        self.device = torch.device("cuda:0" if torch.cuda.is_available() 
                                   else "cpu")
        self.init_net()
        self.net = self.net.to(self.device)
        
        (_, self.image_loader) = load_test_stimuli(self.data_dir)
        
    def init_net(self):
        if self.net_name == "vgg16":
            self.net = models.vgg16(pretrained=self.pretrained)
        elif self.net_name == "alexnet":
            self.net = models.alexnet(pretrained=self.pretrained)
        else:
            # default to vgg16
            self.net = models.vgg16(pretrained=self.pretrained)
            
        # update net's output layer to match n_classes
        n_features = self.net.classifier[-1].in_features
        self.net.classifier[-1] = nn.Linear(n_features, self.n_classes)
    
    def save_net_snapshot(self, epoch):
        print(f"Saving network snapshot at epoch {epoch}")
        
        net_tag = get_net_tag(self.net_name, epoch)
        filename = f"{net_tag}.pt"
        net_output_dir = os.path.join(self.data_dir, "nets/")
        net_filepath = os.path.join(net_output_dir, filename)
        
        snapshot_state = {
            "epoch": epoch,
            "state_dict": self.net.state_dict()
        }
        torch.save(snapshot_state, net_filepath)
    
    def load_net_snapshot(self, epoch):
        self.epoch = epoch
        net_tag = get_net_tag(self.net_name, self.epoch)
        filename = f"{net_tag}.pt"
        net_output_dir = os.path.join(self.data_dir, "nets/")
        net_filepath = os.path.join(net_output_dir, filename)
        
        self.init_net()
        snapshot_state = torch.load(net_filepath, map_location=self.device)
        # null check for backward compatibility
        state_dict = snapshot_state.get("state_dict") if snapshot_state.get("state_dict") is not None else snapshot_state
        self.net.load_state_dict(state_dict)
        self.net = self.net.to(self.device)
        self.net.eval()
        
    def save_net_responses(self):
        self._responses = torch.stack(self._responses)
        net_tag = get_net_tag(self.net_name, self.epoch)
        filename = f"{net_tag}_{self.target_layer}_output.pt"
        sub_dir = f"net_responses/{self.net_name}/"
        output_dir = os.path.join(self.data_dir, sub_dir)
        output_filepath = os.path.join(output_dir, filename)
        torch.save(self._responses, output_filepath)
        
    def set_output_hook(self, target_layer):
        # store responses here...
        self._responses = []
        self.target_layer = target_layer
        
        # define hook fn
        def _hook(module, inp, output):
            self._responses.append(output)
        
        # just hook up single layer for now
        i_layer = nets[self.net_name]["layers_of_interest"][target_layer]
        self.net.features[i_layer].register_forward_hook(_hook)
        
        # set ReLU layer in place rectification to false to get unrectified responses
        potential_relu_layer = self.net.features[i_layer + 1]
        if isinstance(potential_relu_layer, nn.ReLU):
            print("Setting inplace rectification to false!")
            potential_relu_layer.inplace = False
            
    def run_test_stimuli(self, image_loader):
        self._responses = []
        
        for i_im, data in enumerate(image_loader):
            imgs, labels = data          
            self.net(imgs)
        
        print(f"Generated {len(self._responses)} responses.")
        
    def load_imagenette(self):
        data_transforms = {
            "train": transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(), 
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            "val": transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        
        imagenette_dir = os.path.join(self.data_dir, "imagenette2/")
        image_datasets = { x: datasets.ImageFolder(os.path.join(imagenette_dir, x),
                                                   data_transforms[x])
                          for x in ["train", "val"] }
        
        self.dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                     shuffle=True, num_workers=4)
                          for x in ['train', 'val']}
        
        self.dataset_sizes = { x: len(image_datasets[x]) for x in ["train", "val"]}
        self.class_names = image_datasets["train"].classes
        self.n_classes = len(self.class_names)
        
    def evaluate_model(self, criterion, optimizer):
        self.net.eval()
        
        phase = "val"
        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for inputs, labels in self.dataloaders[phase]:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(False):
                outputs = self.net(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / self.dataset_sizes[phase]
        epoch_acc = running_corrects.double() / self.dataset_sizes[phase]

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            phase, epoch_loss, epoch_acc))
        
    
    def train_model(self, criterion, optimizer, scheduler, n_epochs=25, n_snapshots=4):
        since = time.time()
        
        best_model_wts = copy.deepcopy(self.net.state_dict())
        best_acc = 0.0
        best_epoch = -1
    
        epochs = range(self.epoch, self.epoch + n_epochs)
        for epoch in epochs:
            print('Epoch {}/{}'.format(epoch, n_epochs - 1))
            print('-' * 10)
    
            # check if we should take a snapshot
            if (epoch % math.ceil(n_epochs/n_snapshots) == 0):
              self.save_net_snapshot(epoch)
    
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.net.train()  # Set net to training mode
                else:
                    self.net.eval()   # Set net to evaluate mode
    
                running_loss = 0.0
                running_corrects = 0
    
                # Iterate over data.
                for inputs, labels in self.dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
    
                    # zero the parameter gradients
                    optimizer.zero_grad()
    
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.net(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
    
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
    
                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()
    
                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = running_corrects.double() / self.dataset_sizes[phase]
    
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))
    
                # deep copy the net
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_epoch = epoch
                    best_model_wts = copy.deepcopy(self.net.state_dict())
    
            print()
    
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f} on epoch {}'.format(best_acc, best_epoch))
    
        # save snapshot at end (not necessarily best...)
        self.save_net_snapshot(epoch)

if __name__ == "__main__":
    net_manager = NetManager("vgg16", 10, "data")
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
              


