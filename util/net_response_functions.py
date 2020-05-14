# -*- coding: utf-8 -*-

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