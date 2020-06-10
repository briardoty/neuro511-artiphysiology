#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 16:49:58 2020

@author: briardoty
"""
import numpy as np
import torch
import xarray as xr
from torchvision import datasets, models, transforms
import os
from NetManager import get_net_tag
from NetResponseProcessor import *
from ActivationPatches import v227
from NetManager import *
import math
import matplotlib.pyplot as plt
import pickle


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

class ImageNetteVisualizer(NetResponseProcessor):
    
    def __init__(self, net_name, trial, layer_name, net_snapshot, data_dir):
        
        # load all the typical stuff from NetResponseProcessor
        super().__init__(net_name, trial, layer_name, net_snapshot, data_dir)
        
        # also load imagenette
        (self.image_datasets,
         self.train_loader, 
         self.val_loader, 
         _, 
         _) = load_imagenette(self.data_dir)
        
    def show_top_and_bottom_with_box(self, ti, tr, tc, bi, br, bc, rfpd, k):
        """

        Parameters
        ----------
        ti : [units][10]
            Top image indices.
        tr : [units][10]
            Top responses.
        tc : [units][10]
            Top spatial coord tuples.
        bi : [units][10]
            Bottom image indices.
        br : [units][10]
            Bottom responses.
        bc : [units][10]
            tuples of bottom spatial coords.
        rfpd : dict
            RF parameter dict.
        k : int
            Unit.

        """
        sz = rfpd['size']
        x0 = rfpd['x0']
        dx = rfpd['stride']
        print("  rfdp (sz, x0, dx):",sz,x0,dx)
        
        n = len(ti[k])
        for i in range(n):
            j = ti[k][i]
            print("   unit_xy= ",tc[k][i])
            i0 = x0 + dx * tc[k][i][0]
            j0 = x0 + dx * tc[k][i][1]
            print("  Top",i,"image",j," response =",tr[k][i])
            title = "Unit " + str(k) + "  Top " + str(i+1)
            self.im_show_box(j,i0,j0,sz,title)
        
        for i in range(n):
            j = bi[k][i]
            i0 = x0 + dx * bc[k][i][0]
            j0 = x0 + dx * bc[k][i][1]
            print("  Bottom",i,"image",j," response =",br[k][i])
            title = "Unit " + str(k) + "  Bot " + str(i+1)
            self.im_show_box(j,i0,j0,sz,title)
            
    def im_show_box(self, k, i0, j0, w, title):
        #
        #       k  - Index of image
        #  (i0,j0) - lower left corner of box (pix)
        #       w  - width of box (pix)
        #
        
        # todo: this only works with train OR val set now...
        d, _ = self.image_datasets["train"][k]
        d = d.numpy()
        
        dmin = d.min()
        dmax = d.max()
        d = d - dmin
        d = d / (dmax - dmin)
        self.im_draw_box(d,i0,j0,w)
        plt.imshow(np.transpose(d, (1,2,0)))
        plt.title(title)
        plt.show()
        
    def im_draw_box(self, d,i0,j0,w):
        #      d  - numpy array [3][xn][xn] to over-write
        # (i0,j0) - initial point (pix)
        #      w  - size of box (pix)
        print("     box  (",i0,",",j0,")  wid",w)
        xn = len(d[0])
        
        for i in range(i0,i0+w):
            if (i >= 0) & (i <  xn):
                if (i == i0) | (i == i0+w-1):
                    for j in range(j0,j0+w):
                        if (j >= 0) & (j <  xn):
                            d[0][i][j] = 1.0
                            d[1][i][j] = 0.0
                            d[2][i][j] = 0.0
                else:
                    if (j0 >= 0) & (j0 <  xn):
                        d[0][i][j0] = 1.0
                        d[1][i][j0] = 0.0
                        d[2][i][j0] = 0.0
                    if (j0+w-1 >= 0) & (j0+w-1 <  xn):
                        d[0][i][j0+w-1] = 1.0
                        d[1][i][j0+w-1] = 0.0
                        d[2][i][j0+w-1] = 0.0


if __name__ == "__main__":
    # params
    net_name = "vgg16"
    snapshot = 18
    trial = 2
    layer_name = "conv8"
    data_dir = "./data"
    unit = 271
    save_fig = False
    
    # load activation patch data
    sub_dir = os.path.join(data_dir, f"activation_patches/{net_name}/trial{trial}/")
    net_tag = get_net_tag(net_name, snapshot)
    
    top_r = np.load(os.path.join(sub_dir, f"{net_tag}_{layer_name}_t10_r.npy"))
    top_i = np.load(os.path.join(sub_dir, f"{net_tag}_{layer_name}_t10_i.npy"))
    bot_r = np.load(os.path.join(sub_dir, f"{net_tag}_{layer_name}_b10_r.npy"))
    bot_i = np.load(os.path.join(sub_dir, f"{net_tag}_{layer_name}_b10_i.npy"))
    
    with open(os.path.join(sub_dir, f"{net_tag}_{layer_name}_t10_c.txt"),"rb") as fp:
        top_c = pickle.load(fp)
    
    with open(os.path.join(sub_dir, f"{net_tag}_{layer_name}_b10_c.txt"),"rb") as fp:
        bot_c = pickle.load(fp)
    
    # visualize
    visualizer = ImageNetteVisualizer(net_name, trial, layer_name, snapshot, data_dir)
    rfpd = v227[layer_name]
    visualizer.show_top_and_bottom_with_box(
        top_i, top_r, top_c, bot_i, bot_r, bot_c, rfpd, 0)

