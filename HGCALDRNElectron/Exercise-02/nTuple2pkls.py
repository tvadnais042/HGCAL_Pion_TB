#!/usr/bin/env python3
import os
import argparse

import numpy as np
import uproot
import awkward as ak
import pandas as pd
from time import time

import torch
from torch_geometric.data import Data
import pickle

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.patches import RegularPolygon
from matplotlib.colors import to_rgb, to_rgba

# Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--path_to_input', type=str, default='/home/rusack/shared/nTuples/hgcal_electrons/data/ntuple_653.root')
parser.add_argument('-t', '--input_type', type=str, default='file')
parser.add_argument('-o', '--output_folder', type=str, default='/home/rusack/shared/pickles/test/')
parser.add_argument('-d', '--data_type', type=str, default='data')
parser.add_argument('--mc', action='store_true')
args = parser.parse_args()

filelist = []

if args.input_type=='file':
    filelist.append(args.path_to_input)
elif args.input_type=='folder':
    filelist = [ args.path_to_input+f.replace('\\','') for f in os.listdir(args.path_to_input) if '.root' in f ]
else:
    print('Invalid file type!')
    exit(2)

# check if files exist
for f in filelist:
    if not os.path.exists(f):
        print('File not found!')
        exit(2)

########################################
#           HGCAL Values               #
########################################

HGCAL_X_Min = -18
HGCAL_X_Max = 18

HGCAL_Y_Min = -18
HGCAL_Y_Max = 18

HGCAL_Z_Min = 13
HGCAL_Z_Max = 60 

HGCAL_Min = 0
HGCAL_Max = 2727


def torchify(feat, graph_x = None):
    data = [Data(x = torch.from_numpy(ak.to_numpy(ele).astype(np.float32))) for ele in feat]
    if graph_x is not None:
        for d, gx in zip(data, graph_x):
            d.graph_x = gx
    return data

def rescale(feature, minval, maxval):
    top = feature-minval
    bot = maxval-minval
    return top/bot

def cartfeat_HGCAL(x, y, z, En):
    E = rescale(En, HGCAL_Min, HGCAL_Max)
    x = rescale(x, HGCAL_X_Min, HGCAL_X_Max)
    y = rescale(y, HGCAL_Y_Min, HGCAL_Y_Max)
    z = rescale(z, HGCAL_Z_Min, HGCAL_Z_Max)
    return ak.concatenate((x[:,:,None], y[:,:,None], z[:,:,None], E[:,:,None]), -1)

# set custom functions
plt.rcParams['axes.linewidth'] = 1.4
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['ytick.minor.size'] = 2.5
plt.rcParams['xtick.minor.size'] = 4.0
plt.rcParams['ytick.minor.visible'] = True
plt.rcParams['ytick.labelsize'] = 'large'
plt.rcParams['xtick.labelsize'] = 'large'

def set_custom_alpha(col_, alpha_):
    rgb_ = to_rgba(col_)
    return (col_[0], col_[1], col_[2], alpha_)

def rgb2rgba(col_):
    _ = []
    for c in col_:
        _.append(float(c)/255.0)
    _.append(1.0)
    return tuple(_)

def getNcols(N=3, cmap_='plasma'):
    cmap = plt.get_cmap(cmap_)
    cols = cmap.colors
    arr = []
    for i in range(N):
        arr.append(cols[int(256*float(i)/float(N))])
    return arr

def get_shower_energy_param(hit_map, ip_map):
    sum_e = hit_map.groupby('event')['rechit_energy'].sum()
    ch_e = hit_map[hit_map['rechit_layer']>28].groupby('event')['rechit_energy'].sum()
    ce_e = hit_map[hit_map['rechit_layer']<29].groupby('event')['rechit_energy'].sum()
    ch_n = hit_map[hit_map['rechit_layer']>28].groupby('event')['rechit_energy'].size()
    ce_n = hit_map[hit_map['rechit_layer']<29].groupby('event')['rechit_energy'].size()
    f_eoverh = ce_e/(ce_e+ch_e)

    sum_e.name = 'sum_e'
    f_eoverh.name = 'f_eoverh'

    tmp = pd.DataFrame()
    tmp['sum_e'] = sum_e
    tmp['f_eoverh'] = f_eoverh
    tmp['ch_e'] = ch_e
    tmp['ce_e'] = ce_e
    tmp['ce_n'] = ce_n
    tmp['ch_n'] = ch_n

    tmp.index.name='event'
    hit_map = hit_map.set_index('event').join(tmp)
    hit_map = hit_map.join(ip_map.set_index('event'))

    return hit_map

radius = .5626265648752451

hit_list = ['event', 'rechit_x', 'rechit_y', 'rechit_z', 'rechit_layer',
        'rechit_energy', 'rechit_chip', 'rechit_channel', 'trueBeamEnergy']
ip_list = ['event', 'ntracks', 'dwcReferenceType', 'b_x',
        'b_y', 'm_x', 'm_y', 'trackChi2_X', 'trackChi2_Y']

pickles = {}

t0 = time()

data_rechit_tree = None
data_ip_tree = None
data_hit_map = None
data_ip_map = None

for ifile, f in enumerate(filelist):
    print("Reading file {}".format(f))
    with uproot.open(f) as data_tree:
       data_rechit_tree = data_tree['rechitntupler/hits']
       data_ip_tree = data_tree['trackimpactntupler/impactPoints']
       data_hit_map = data_rechit_tree.arrays(hit_list)
       data_ip_map = data_ip_tree.arrays(ip_list)
       print('File read in {} s'.format(time()-t0))
       print('Applying preselection ...')
       hits_e = data_hit_map.rechit_energy[data_hit_map.rechit_layer<29]
       hits_h = data_hit_map.rechit_energy[data_hit_map.rechit_layer>28]
       sum_e = ak.sum(hits_e, axis=1)
       sum_h = ak.sum(hits_h, axis=1)
       n_e = ak.count(hits_e, axis=1)
       n_h = ak.count(hits_h, axis=1)
       f_eoverh = sum_e/(sum_e+sum_h)
  
       preselection = ak.ones_like(data_hit_map.rechit_x)
  
       data_hit_map['event'] = preselection*data_ip_map.event
       data_hit_map['ce_e'] = preselection*sum_e
       data_hit_map['ch_e'] = preselection*sum_h
       data_hit_map['ce_n'] = preselection*n_e
       data_hit_map['ch_n'] = preselection*n_h
       data_hit_map['f_eoverh'] = preselection*f_eoverh
       data_hit_map['ntracks'] = preselection*data_ip_map.ntracks
       data_hit_map['all_target'] = data_hit_map.trueBeamEnergy*preselection
       dwc_offset_x = data_ip_map.b_x
       dwc_offset_y = data_ip_map.b_y
  
       inf_mask = np.logical_and(np.abs(dwc_offset_x)<100, np.abs(dwc_offset_y)<100)
  
       preselection = ak.ones_like(data_hit_map.rechit_x)

       if args.data_type=='data':
          data_hit_map['offset_b_x'] = (-data_ip_map.b_x + ak.mean(dwc_offset_x[inf_mask]))*preselection
          data_hit_map['offset_b_y'] = (-data_ip_map.b_y + ak.mean(dwc_offset_y[inf_mask]))*preselection
       elif args.data_type=='mc':
          data_hit_map['offset_b_x'] = data_ip_map.b_x-33.2
          data_hit_map['offset_b_y'] = data_ip_map.b_y-20.6
       else:
          print('Invalid data type!')
          exit(2)
       dwc_window_mask = np.logical_and(np.abs(data_hit_map.offset_b_x)<1.0,
               np.abs(data_hit_map.offset_b_y)<1.0)
  
       # define set of preselections
       preselection = ak.ones_like(data_hit_map.rechit_x)
       channel_selections = [np.logical_or(data_hit_map.rechit_chip!=0, data_hit_map.rechit_layer!=1),
               np.logical_or(data_hit_map.rechit_chip!=3, data_hit_map.rechit_channel!=22),
               #        data_hit_map.rechit_layer!=36,
               #        data_hit_map.rechit_layer!=37,
               data_hit_map.rechit_layer<29,
               data_hit_map.rechit_energy>=0.5,
               data_hit_map.ntracks==1,
               dwc_window_mask,
               data_hit_map.f_eoverh>0.95,
               data_hit_map.ch_e<50
               ]
  
       for isel, sel in enumerate(channel_selections):
           print("Selection "+str(isel+1), end=': ')
           preselection = np.logical_and(sel, preselection)
           A = ak.mean(data_hit_map['event'][preselection], axis=1)
           print(len(A[~ak.is_none(A)]))
  
       quantities = ['rechit_x', 'rechit_y', 'rechit_z', 'rechit_energy', 'all_target']
  
       for q_ in quantities:
           if ifile==0:
               if q_=='all_target':
                   pickles[q_] = ak.mean(data_hit_map[q_][preselection], axis=1)[~ak.is_none(A)]
               else:
                   pickles[q_] = data_hit_map[q_][preselection][~ak.is_none(A)]
           else:
               if q_=='all_target':
                   pickles[q_] = np.concatenate((ak.mean(data_hit_map[q_][preselection], axis=1)[~ak.is_none(A)], pickles[q_]), axis=0)
               else:
                   pickles[q_] = np.concatenate((data_hit_map[q_][preselection][~ak.is_none(A)], pickles[q_]), axis=0)
            
arr = cartfeat_HGCAL(pickles['rechit_x'],
               pickles['rechit_y'],
               pickles['rechit_z'],
               pickles['rechit_energy']) 
  
pickles['cartfeat'] = torchify(arr)

print('Dumping pickle files...')
path_to_output = args.output_folder

for q_ in pickles:
    output_file = '{}/{}.pickle'.format(path_to_output, q_)
    with open(output_file, 'wb') as f_:
        t0=time()
        if q_=='cartfeat':
            torch.save(pickles[q_], f_, pickle_protocol=4)
        else:
            pickle.dump(pickles[q_], f_, protocol=4)
        print('File {}.pickle dumped in {} s'.format(q_, time()-t0))
