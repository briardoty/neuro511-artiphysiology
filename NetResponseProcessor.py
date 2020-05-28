#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 17:42:10 2020

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
from NetManager import *
import multiprocessing as mp
from scipy.stats import kurtosis
import tqdm
import xarray as xr


class NetResponseProcessor():
    
    def __init__(self, net_name, layer_name, net_snapshot, data_dir):
        self.net_name = net_name
        self.net_tag = get_net_tag(self.net_name, net_snapshot)
        self.layer_name = layer_name
        self.data_dir = os.path.expanduser(data_dir)
        
        self._load_net_responses(layer_name)
            
        self.apc_models = xr.open_dataset(os.path.join(self.data_dir, 
                                                       "apc_fit/apc_models_362_16x16.nc"))
        
    def _load_net_responses(self, layer_name):
        sub_dir = f"net_responses/{self.net_name}/"
        filename = f"{self.net_tag}_{layer_name}_output.pt"
        output_dir = os.path.join(self.data_dir, sub_dir)
        output_filepath = os.path.join(output_dir, filename)
        
        self._responses_tt = torch.load(output_filepath)
        
    def get_unit_responses(self, units=None):
        """
        Parameters
        ----------
        units : TYPE, optional
            Indices of units to get responses for. The default is None,
            in which case will return all.
    
        Returns
        -------
        unit_responses : TYPE
            DESCRIPTION.
    
        """
        spatial_idx = int((len(self._responses_tt[0, 0, 0]) - 1) / 2)
        
        if units is None:
            unit_responses = self._responses_tt[:, 0, :, spatial_idx, spatial_idx]
        else:
            unit_responses = self._responses_tt[:, 0, units, spatial_idx, spatial_idx]
        
        unit_responses = unit_responses.detach()
        
        return unit_responses
        
    def apc_fit_unit(self, unit):
        """
    
        Parameters
        ----------
        unit : int
            Index of unit to perform APC fit upon.
    
        Returns
        -------
        unit : int
            Index of CNN unit.
        best_model_i : int
            Index of best APC model.
        best_corr : float
            Max correlation coefficient of given unit's actual responses 
            across stimuli and predicted responses of best APC model.
            
        """
        # determine actual responses of given unit to all stimuli
        unit_responses = self.get_unit_responses(unit)
          
        best_corr = np.NINF
        best_model_i = -1
        for i_model in range(len(self.apc_models.models)):
          
            # determine predicted responses of a model to all stimuli
            model_responses = self.apc_models.resp[:, i_model]
            
            # compute correlation
            corr = np.corrcoef(unit_responses, model_responses)[0,1]
            if (corr > best_corr):
                # update best
                best_corr = corr
                best_model_i = i_model
          
        return (unit, best_model_i, best_corr)
        
    def apc_fit_all(self):
        # apply kurtosis filter
        all_unit_responses = self.get_unit_responses()
        unit_kurtosis = kurtosis(all_unit_responses, axis=0)    
        k_filtered_i, = np.where((unit_kurtosis >= 2.9) & (unit_kurtosis <= 42))
        
        n_units = len(k_filtered_i)
        print(f"{n_units} units pass kurtosis filter.")
        with mp.Pool(processes=6) as pool:
            results = list(tqdm.tqdm(pool.imap(self.apc_fit_unit, k_filtered_i),
                                     total=n_units))
        
        output_filename = f"{self.net_tag}_{self.layer_name}_apc_fits.npy"
        output_dir = os.path.join(self.data_dir, f"apc_fit/{net_name}/")
        output_filepath = os.path.join(output_dir, output_filename)
        np.save(output_filepath, results)
    

if __name__ == "__main__":
    net_resp_proc = NetResponseProcessor("vgg16", "conv8", "ep14", "data")
    net_resp_proc.apc_fit_all()
    
    
    
    
    
    
    
    
    
    
    
