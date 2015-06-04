#!/usr/bin/env python3
import numpy as np
import E200
import ipdb
import logging
import matplotlib.pyplot as plt
import pytools as pt
import time
import cProfile

plt.ion()

mylogger = pt.mylogger('test3')
mylogger.setLevel(logging.INFO)

# data = E200.E200_load_data_gui()
data = E200.E200_load_data('nas/nas-li20-pm00/E217/2015/20150603/E217_17827/E217_17827.mat')

imgstr = data.rdrill.data.raw.images.AX_IMG1

# uids = imgstr.UID[0:10]
# imgs = E200.E200_load_images(imgstr, uids)
# mylogger.setLevel(logging.DEBUG)

uids_wanted = imgstr.UID
uids_wanted = uids_wanted[uids_wanted > 1e5]


def myloop():
    for i, img in enumerate(E200.E200_Image_Iter(imgstr, numperset=100, uids=uids_wanted)):
        print('Iteration: {}'.format(i))

# myloop()
cProfile.run('myloop()', sort='tottime')
# ipdb.set_trace()
