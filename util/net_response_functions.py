# -*- coding: utf-8 -*-
import numpy as np

def get_unit_responses(outputs_tt, units=None):
    """
    Parameters
    ----------
    outputs_tt : TYPE
        DESCRIPTION.
    units : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    unit_responses : TYPE
        DESCRIPTION.

    """
    spatial_idx = int((len(outputs_tt[0, 0, 0]) - 1) / 2)
    
    if units is None:
        unit_responses = outputs_tt[:, 0, :, spatial_idx, spatial_idx]
    else:
        unit_responses = outputs_tt[:, 0, units, spatial_idx, spatial_idx]
    
    unit_responses = unit_responses.detach()
    
    return unit_responses

def get_n_most_selective_units(n, apc_fits, apc_models, outputs_tt):
    """
    Parameters
    ----------
    n : TYPE
        DESCRIPTION.
    apc_fits : TYPE
        DESCRIPTION.
    apc_models : TYPE
        DESCRIPTION.
    outputs_tt : TYPE
        DESCRIPTION.

    Returns
    -------
    Array
        N units with highest correlation to APC model across stimuli.

    """
    
    # apply kurtosis filter
    # unit_responses[stim][unit]
    #unit_responses = get_unit_responses(outputs_tt)
    #unit_kurtosis = kurtosis(unit_responses, axis=0)    
    #k_filtered_i, = np.where((unit_kurtosis >= 2.9) & (unit_kurtosis <= 42))
    
    # get most selective units from those that pass kurtosis filter
    #k_filtered_fits = apc_fits[k_filtered_i]
    sorted_fits = np.argsort(apc_fits[:,2])
    sorted_fits = apc_fits[sorted_fits]
    
    return sorted_fits[::-1][:n]
