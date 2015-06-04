#!/usr/bin/env python3
import ipdb
import E200
from classes import *  # NOQA
# import os
# import PIL
import numpy as np
import matplotlib.pyplot as plt  # NOQA

data = E200.E200_load_data(filename='nas/nas-li20-pm00/E217/2015/20150603/E217_17828/E217_17828.mat')

imgstr = data.rdrill.data.raw.images.AX_IMG1

cam = 'AX_IMG1'
cal = 10e-6

# ======================================
# UIDs in common
# ======================================
camlist      = ['AX_IMG1', 'AX_IMG2']
calibrations = [10e-6, 17e-6]
uids = np.empty(2, dtype=object)
for i, cam in enumerate(camlist):
    imgstr_temp  = getattr(data.rdrill.data.raw.images, cam)
    uids[i] = imgstr_temp.UID

uids_wanted = np.intersect1d(uids[0], uids[1])
uids_wanted = uids_wanted[uids_wanted > 1e5]

for i, (cam, cal) in enumerate(zip(camlist, calibrations)):
    imgstr = getattr(data.rdrill.data.raw.images, cam)
    blob = BlobAnalysis(imgstr, imgname=cam, cal=cal, reconstruct_radius=1, uids=uids_wanted)
    this = np.append(blob.centroid.transpose(), [blob._timestamps], axis=0)
    np.savetxt('series_{}.csv'.format(cam), this.transpose(), delimiter=', ')
