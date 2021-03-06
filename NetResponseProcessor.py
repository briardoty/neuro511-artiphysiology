#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 17:42:10 2020

@author: briardoty
"""
import torch
import numpy as np
import os
import multiprocessing as mp
from scipy.stats import kurtosis
import tqdm
from NetManager import get_net_tag
import xarray as xr


class NetResponseProcessor():
    
    def __init__(self, net_name, trial, layer_name, net_snapshot, data_dir):
        self.net_name = net_name
        self.trial = trial
        self.net_tag = get_net_tag(self.net_name, net_snapshot)
        self.layer_name = layer_name
        self.data_dir = os.path.expanduser(data_dir)
        
        self.load_net_responses()
            
        self.apc_models = xr.open_dataset(os.path.join(self.data_dir, 
                                                       "apc_fit/apc_models_362_16x16.nc"))
        
    def load_net_responses(self):
        sub_dir = os.path.join(self.data_dir, f"net_responses/{self.net_name}/")
        if self.trial is not None:
            sub_dir = os.path.join(sub_dir, f"trial{self.trial}/")
        
        filename = f"{self.net_tag}_{self.layer_name}_output.pt"
        output_filepath = os.path.join(sub_dir, filename)
        
        self.responses_tt = torch.load(output_filepath)
        
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
        spatial_idx = int((len(self.responses_tt[0, 0, 0]) - 1) / 2)
        
        if units is None:
            unit_responses = self.responses_tt[:, 0, :, spatial_idx, spatial_idx]
        else:
            unit_responses = self.responses_tt[:, 0, units, spatial_idx, spatial_idx]
        
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
        
    def apc_fit_all(self, processes=4):
        # apply kurtosis filter
        all_unit_responses = self.get_unit_responses()
        unit_kurtosis = kurtosis(all_unit_responses, axis=0)    
        k_filtered_i, = np.where((unit_kurtosis >= 2.9) & (unit_kurtosis <= 42))
        
        n_units = len(k_filtered_i)
        print(f"{n_units} units pass kurtosis filter.")
        with mp.Pool(processes=processes) as pool:
            results = list(tqdm.tqdm(pool.imap(self.apc_fit_unit, k_filtered_i),
                                     total=n_units))
        
        output_filename = f"{self.net_tag}_{self.layer_name}_apc_fits.npy"
        output_dir = os.path.join(self.data_dir, f"apc_fit/{self.net_name}/")
        output_filepath = os.path.join(output_dir, output_filename)
        np.save(output_filepath, results)
    

if __name__ == "__main__":
    net_resp_proc = NetResponseProcessor("vgg16", "conv8", "ep14", "data")
    net_resp_proc.apc_fit_all(6)
    
    
    
    
    
    
    
    
    
    
    
