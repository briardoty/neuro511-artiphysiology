#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 13:42:29 2020

@author: briardoty
"""
from util.net_response_functions import *
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, models, transforms
import math


def plot_apc_fits_for_unit(net_name, layer_name, unit, apc_fits, apc_models, outputs_tt, save_fig=False):
    """
    
    Parameters
    ----------
    net_name : TYPE
        DESCRIPTION.
    layer_name : TYPE
        DESCRIPTION.
    unit : TYPE
        DESCRIPTION.
    apc_fits : TYPE
        DESCRIPTION.
    apc_models : TYPE
        DESCRIPTION.
    outputs_tt : TYPE
        DESCRIPTION.
    save_fig : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    """
    
    # convert to dict
    apc_fits_dict = {unit : (i_model, corr) for (unit, i_model, corr) in apc_fits}
    
    # get x data (actual unit responses across stim)
    unit_responses = get_unit_responses(outputs_tt, unit)
    
    # get y data (predicted responses from unit's best fit model)
    (i_model, best_corr) = apc_fits_dict[unit]
    # print(f"Correlation: {best_corr}")
    model_responses = apc_models.resp[:, int(i_model)]
    
    # apc params
    model = apc_models.models[int(i_model)]
    mu_a = math.degrees(model.or_mean.values)
    sd_a = math.degrees(model.or_sd.values)
    mu_c = model.cur_mean.values
    sd_c = model.cur_sd.values
    
    # plot 'em
    fig, axes = plt.subplots(1, figsize=(5, 5))
    title = f"{net_name}_{layer_name}_u{unit}"
    fig.suptitle(title)
    axes.set_title(f"mu_a={round(mu_a, 3)}; mu_c={mu_c}; corr={round(best_corr, 3)}")
    axes.set_xlabel("Unit response")
    axes.set_ylabel("Model prediction")
    
    axes.plot(unit_responses, model_responses, "k.")
    
    # optional save
    if not save_fig:
        return

    name = f"./data/figures/{net_name}/{title}.png"
    plt.savefig(name, dpi=300, bbox_inches='tight')

def imshow(img):
    """
    Denormalize and display the given image

    Parameters
    ----------
    img : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    img = (img - img.min())/(img - img.min()).max()
    plt.imshow(np.transpose(img, (1, 2, 0))) # convert from Tensor image

def show_images(imgs, idxs):
    """
    Plot images idxs from the given ImageFolder
    

    Parameters
    ----------
    imgs : TYPE
        DESCRIPTION.
    idxs : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    fig = plt.figure(figsize=(25, 4))

    fig_i = 0
    for i in idxs:
        img, label = imgs[i]
        img = img.numpy()
        ax = fig.add_subplot(2, 20/2, fig_i+1, xticks=[], yticks=[])
        imshow(img)
        fig_i += 1

def display_top_and_bottom(unit, n, outputs_tt):
    """
    For a given unit, display the top/bottom n stimuli ranked by response activation
    
    Parameters
    ----------
    unit : TYPE
        DESCRIPTION.
    n : TYPE
        DESCRIPTION.
    outputs_tt : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    # pull in images
    img_xy = 200
    data_transforms = transforms.Compose([
        transforms.CenterCrop(img_xy),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225])
    ])
    
    image_dataset = datasets.ImageFolder(
        root="./data/stimuli",
        transform=data_transforms)
    
    image_loader = torch.utils.data.DataLoader(image_dataset)

    # get all responses for given unit
    unit_outputs = get_unit_responses(outputs_tt, unit)

    # get indices top/bottom n stimuli
    top_n = torch.topk(unit_outputs, k=n)
    bot_n = torch.topk(unit_outputs, k=n, largest=False)

    top_n_idx = top_n.indices
    bot_n_idx = bot_n.indices

    # display those stimuli
    show_images(image_dataset, top_n_idx)
    show_images(image_dataset, bot_n_idx)
    
    
    
    
    
    
    
    
    
    
    
