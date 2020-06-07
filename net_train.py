#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 13:27:35 2020

@author: briardoty
"""
import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import os
import time
from util.visualization_functions import *


##########################
#                        #
#        params          #
#                        #
##########################
net_name = "vgg16"
snapshots = [7, 21]
layer_name = "conv8"
data_dir = "data"

##########################
#                        #
#       initialize       #
#                        #
##########################
manager = NetManager(net_name, 10, data_dir, pretrained=False)

criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(manager.net.parameters(), lr=0.05, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# manager.load_net_snapshot(33)
manager.load_imagenette()

##########################
#                        #
#     training a net     #
#                        #
##########################
manager.train_model(criterion, optimizer, exp_lr_scheduler, n_epochs=20, n_snapshots=4)

















