#!/usr/bin/env python3
import ipdb
import E200
from classes import *  # NOQA
# import os
# import PIL
import numpy as np
import matplotlib.pyplot as plt  # NOQA

data = E200.E200_load_data(filename='nas/nas-li20-pm00/E217/2015/20150603/E217_17828/E217_17828.mat')

# ======================================
# UIDs in common
# ======================================
cam    = 'LRoomFar'
cal    = 1
imgstr = getattr(data.rdrill.data.raw.images, cam)
uids   = imgstr.UID

uids_wanted = uids[uids > 1e5]

blob = BlobAnalysis(imgstr, imgname=cam, cal=cal, reconstruct_radius=1, uids=uids_wanted, movie=True)
# this = np.append(blob.centroid.transpose(), [blob._timestamps], axis=0)
# np.savetxt('series_{}.csv'.format(cam), this.transpose(), delimiter=', ')
# ipdb.set_trace()
fig = fft(blob
