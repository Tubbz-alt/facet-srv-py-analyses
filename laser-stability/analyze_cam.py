#!/usr/bin/env python3
import argparse
import ipdb  # NOQA
import E200
from classes import *  # NOQA
# import os
# import PIL
import numpy as np  # NOQA
import scipy as sp  # NOQA
import matplotlib.pyplot as plt  # NOQA
import logging
logger = logging.getLogger(__name__)
from fft import *  # NOQA


def fft_analyze_cam(camlist, cal_list, filename=None):
    if filename is None:
        # data = E200.E200_load_data(filename='nas/nas-li20-pm00/E200/2015/20150602/E200_17712/E200_17712.mat')
        data = E200.E200_load_data_gui()
    else:
        data = E200.E200_load_data(filename)

    num_cams = np.size(camlist)
    blobs = np.empty(num_cams, dtype=object)

    # ipdb.set_trace()

    for i, (cam, cal) in enumerate(zip(camlist, cal_list)):
        # ======================================
        # UIDs in common
        # ======================================
        imgstr = getattr(data.rdrill.data.raw.images, cam)
        uids   = imgstr.UID
        
        uids_wanted = uids[uids > 1e5]
        
        blobs[i] = BlobAnalysis(imgstr, imgname=cam, cal=cal, reconstruct_radius=1, uids=uids_wanted)
        # this = np.append(blob.centroid.transpose(), [blob._timestamps], axis=0)
        # np.savetxt('series_{}.csv'.format(cam), this.transpose(), delimiter=', ')
        # ipdb.set_trace()
        
        freq = data.rdrill.data.raw.metadata.E200_state.EVNT_SYS1_1_BEAMRATE.dat[0]

    fig  = fft(blobs, camlist, freq=freq)
    
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=
            'Analyzes laser stability.')
    parser.add_argument('-V', action='version', version='%(prog)s v0.1')
    parser.add_argument('-v', '--verbose', action='store_true',
            help='Verbose mode.')
    parser.add_argument('-s', '--save', action='store_true',
            help='Save movie files')
    parser.add_argument('-d', '--debug', action='store_true',
            help='Open debugger after running')
    parser.add_argument('-f', '--filename',
            help='Dataset filename')
    parser.add_argument('-c', '--camera', action='append',
            help='Camera')
    parser.add_argument('--cal', default=1, action='append', type=float,
            help='Calibration')
    arg = parser.parse_args()

    num_cams = np.size(arg.camera)
    if np.size(arg.cal) == num_cams:
        cal = arg.cal
    else:
        cal_flat = np.array([arg.cal]).flatten()
        cal = np.ones(num_cams) * np.float(cal_flat[0])

    fft_analyze_cam(filename=arg.filename, camlist=arg.camera, cal_list=cal)
