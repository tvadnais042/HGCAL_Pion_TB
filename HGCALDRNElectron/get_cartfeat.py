#!/usr/bin/env python3
import sys
import argparse
import pickle
import json

parser = argparse.ArgumentParser()
parser.add_argument('folder', type=str, default='data/hgcal_electrons/test')
args = parser.parse_args()

folder = args.folder

import torch
from torch_geometric.data import Data
import awkward as ak
import numpy as np

# import input range
with open('utils/hgcal_input_range.json','r') as f:
        input_range = json.load(f)

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
    E = rescale(En, input_range['HGCAL_Min'], input_range['HGCAL_Max'])
    x = rescale(x, input_range['HGCAL_X_Min'], input_range['HGCAL_X_Max'])
    y = rescale(y, input_range['HGCAL_Y_Min'], input_range['HGCAL_Y_Max'])
    z = rescale(z, input_range['HGCAL_Z_Min'], input_range['HGCAL_Z_Max'])
    return ak.concatenate((x[:,:,None], y[:,:,None], z[:,:,None], E[:,:,None]), -1)

pickles = ['rechit_x', 'rechit_y', 'rechit_z', 'rechit_energy']
arrays = {}
for p in pickles:
    filename = '{}/{}.pickle'.format(folder, p)
    print(filename)
    with open(filename, 'rb') as file_:
        arrays[p] = pickle.load(file_)

carray = cartfeat_HGCAL(arrays['rechit_x'],
             arrays['rechit_y'],
             arrays['rechit_z'],
             arrays['rechit_energy'])

print('Produced carray')
carray = torchify(carray)

with open('{}/cartfeat.pickle'.format(folder), 'wb') as file_:
    torch.save(carray, file_, pickle_protocol=4)

print('Torched')
