#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 14:55:27 2020

@author: briardoty
"""

import xarray as xr
import numpy as np
import torch
import multiprocessing as mp
import os
import sys
import tqdm
from scipy.stats import kurtosis
from util.net_response_functions import *

def apc_fit_unit(unit):
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
    unit_responses = get_unit_responses(outputs_tt, unit)
    # print(f"Determined actual responses for {len(unit_responses)} stimuli.")
      
    best_corr = np.NINF
    best_model_i = -1
    for i_model in range(len(apc_models.models)):
      
        # determine predicted responses of a model to all stimuli
        model_responses = apc_models.resp[:, i_model]
        
        # compute correlation
        corr = np.corrcoef(unit_responses, model_responses)[0,1]
        if (corr > best_corr):
            # update best
            best_corr = corr
            best_model_i = i_model
            # print(f"Found new best model with correlation = {best_corr}!")
      
    return (unit, best_model_i, best_corr)

if __name__ == "__main__":
    # parse args
    if len(sys.argv) < 3:
        net_name = "vgg16"
        layer_name = "conv10"        
    else:        
        net_name = sys.argv[1]
        layer_name = sys.argv[2]
    
    # load apc data
    apc_models = xr.open_dataset("data/apc_fit/apc_models_362_16x16.nc")
    
    # load model output
    outputs_tt = torch.load(f"data/net_responses/{net_name}/{net_name}_{layer_name}_output.pt")
    
    # apply kurtosis filter
    all_unit_responses = get_unit_responses(outputs_tt)
    unit_kurtosis = kurtosis(all_unit_responses, axis=0)    
    k_filtered_i, = np.where((unit_kurtosis >= 2.9) & (unit_kurtosis <= 42))
    
    n_units = len(k_filtered_i)
    print(f"{n_units} pass kurtosis filter.")
    with mp.Pool(processes=6) as pool:
        results = list(tqdm.tqdm(pool.imap(apc_fit_unit, k_filtered_i),
                                 total=n_units))
    
    output_filename = f"{net_name}_{layer_name}_apc_fits.npy"
    output_dir = os.path.expanduser(f"data/apc_fit/{net_name}")
    output_filepath = os.path.join(output_dir, output_filename)
    np.save(output_filepath, results)

















