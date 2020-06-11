#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 13:28:39 2020

@author: briardoty
"""
import numpy as np
import torch
import xarray as xr
from torchvision import datasets, models, transforms
import os
from NetResponseProcessor import *
from NetManager import *
import math
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf


def imshow(img):
    """
    Denormalize and display the given image

    Parameters
    ----------
    img : Torch tensor
        Image to display.

    Returns
    -------
    None.

    """
    img = (img - img.min())/(img - img.min()).max()
    plt.imshow(np.transpose(img, (1, 2, 0))) # convert from Tensor image

class Visualizer(NetResponseProcessor):
    
    def __init__(self, net_name, trial, layer_name, net_snapshot, data_dir):
        
        # load all the typical stuff from NetResponseProcessor
        super().__init__(net_name, trial, layer_name, net_snapshot, data_dir)
        
        # also load apc fits
        sub_dir = os.path.join(self.data_dir, f"apc_fit/{self.net_name}/")
        if self.trial is not None:
            sub_dir = os.path.join(sub_dir, f"trial{self.trial}/")
        self.apc_fits = np.load(
            os.path.join(sub_dir, 
                         f"{self.net_tag}_{self.layer_name}_apc_fits.npy"),
            allow_pickle=True
        )
        
        # and test stim for display
        (self.image_dataset, _) = load_test_stimuli(self.data_dir)
    
    def plot_unit_summary(self, unit, save_fig=False):
        apc_fig = self.plot_apc_fits_for_unit(unit)
        (top_fig, bot_fig) = self.display_top_and_bottom(unit, 10)
        
        plt.show()
        
        if not save_fig:
            return
        
        sub_dir = os.path.join(self.data_dir, f"figures/{self.net_name}/")
        if (self.trial is not None):
            sub_dir = os.path.join(sub_dir, f"trial{self.trial}/")
        filename = f"{self.net_tag}_{self.layer_name}_u{unit}.pdf"
        pdf_file = os.path.join(sub_dir, filename)
        pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_file)
        pdf.savefig(apc_fig)
        pdf.savefig(top_fig)
        pdf.savefig(bot_fig)
        pdf.close()
    
    def plot_apc_fits_for_unit(self, unit):
    
        # convert to dict
        apc_fits_dict = {unit : (i_model, corr) 
                         for (unit, i_model, corr) 
                         in self.apc_fits
                         }
        
        # get x data (actual unit responses across stim)
        unit_responses = self.get_unit_responses(unit)
        
        # get y data (predicted responses from unit's best fit model)
        (i_model, best_corr) = apc_fits_dict[unit]
        # print(f"Correlation: {best_corr}")
        model_responses = self.apc_models.resp[:, int(i_model)]
        
        # apc params
        model = self.apc_models.models[int(i_model)]
        mu_a = math.degrees(model.or_mean.values)
        sd_a = math.degrees(model.or_sd.values)
        mu_c = model.cur_mean.values
        sd_c = model.cur_sd.values
        
        # plot 'em
        fig, axes = plt.subplots(1, figsize=(7, 5))
        title = f"{self.net_tag}_{self.layer_name}_u{unit}"
        fig.suptitle(title)
        subtitle = "mu_a={:.2f}; sd_a={:.2f}; mu_c={:.2f}; sd_c={:.2f}; corr={:.4f}".format(
            mu_a, sd_a, mu_c, sd_c, best_corr)
        axes.set_title(subtitle)
        axes.set_xlabel("Unit response")
        axes.set_ylabel("Model prediction")
        
        axes.plot(unit_responses, model_responses, "k.")
        
        return fig
        
    def display_top_and_bottom(self, unit, n):
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
        
        # get all responses for given unit
        unit_outputs = self.get_unit_responses(unit)
    
        # get indices top/bottom n stimuli
        top_n = torch.topk(unit_outputs, k=n)
        bot_n = torch.topk(unit_outputs, k=n, largest=False)
    
        top_n_idx = top_n.indices
        bot_n_idx = bot_n.indices
    
        # display those stimuli
        title = f"{self.net_tag}_{self.layer_name}_u{unit}"
        top_fig = self.show_images(top_n_idx, "TOP " + title)
        bot_fig = self.show_images(bot_n_idx, "BOT " + title)
        
        return (top_fig, bot_fig)
        
    def show_images(self, idxs, title):
        """
        Plot images with indexes idxs from self.image_dataset
        
    
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
        fig = plt.figure(figsize=(15, 3))
        
        fig_i = 1
        for i in idxs:
            img, label = self.image_dataset[i]
            img = img.numpy()
            ax = fig.add_subplot(1, len(idxs), fig_i)
            ax.axis("off")
            fig.suptitle(title)
            imshow(img)
            fig_i += 1
        
        return fig
        
    def get_n_most_selective_units(self, n):
        """
        Parameters
        ----------
        n : int
            Number units to analyze.
    
        Returns
        -------
        Array
            N units with highest correlation to APC model across stimuli.
    
        """
        
        sorted_fits = np.argsort(self.apc_fits[:,2])
        sorted_fits = self.apc_fits[sorted_fits]
        
        return sorted_fits[::-1][:n]
        
    
if __name__ == "__main__":
    net_name = "vgg16"
    snapshot = 24
    layer_name = "conv8"
    data_dir = "./data"
    unit = 271
    save_fig = True
    
    visualizer = Visualizer(net_name, 1, layer_name, snapshot, data_dir)
    visualizer.plot_unit_summary(unit, save_fig)
    
    
    
    
    
    
    
    
    
    
    
    
    