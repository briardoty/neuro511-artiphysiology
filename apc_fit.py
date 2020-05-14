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
    if len(sys.argv) < 2:
        layer_name = "conv10"
    else:        
        layer_name = sys.argv[1]
    
    # load apc data
    apc_models = xr.open_dataset("data/apc_fit/apc_models_362_16x16.nc")
    
    # load model output
    outputs_tt = torch.load(f"data/net_responses/vgg16_{layer_name}_output.pt")
    
    n_units = len(outputs_tt[0,0])
    with mp.Pool(processes=6) as pool:
        results = list(tqdm.tqdm(pool.imap(apc_fit_unit, range(n_units)),
                                 total=n_units))
    
    output_filename = f"vgg16_{layer_name}_apc_fits.npy"
    output_dir = os.path.expanduser("data/apc_fit")
    output_filepath = os.path.join(output_dir, output_filename)
    np.save(output_filepath, results)

















