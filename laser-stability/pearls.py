#!/usr/bin/env python3
import shutil
import os
import argparse
import E200
import h5py as h5  # NOQA
import ipdb                              # NOQA
import matplotlib as mpl                 # NOQA
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pytools as pt
import shlex                             # NOQA
import skimage.feature as skfeat         # NOQA
import skimage.filters as skfilt         # NOQA
import skimage.measure as skmeas
import skimage.morphology as skmorph     # NOQA
import skimage.segmentation as skseg     # NOQA
import skimage.transform as sktrans      # NOQA
import subprocess                        # NOQA
import sys

# verbose = True
verbose = False
reconstruct_radius = 2

gs      = gridspec.GridSpec(1, 1)

rad10 = skmorph.disk(10)


def movie_imshow(fig_mov, ax_mov, image, centroid, filename=None, toplabel='', xlabel='', ylabel='', **kwargs):
    ax_mov.cla()
    ax_mov.imshow(image, cmap=mpl.cm.cool, **kwargs)
    ax_mov.plot(centroid[1], centroid[0], 'ro', scalex=False, scaley=False)

    # pt.addlabel(ax=ax_mov, toplabel=toplabel, xlabel=xlabel, ylabel=ylabel)

    fig_mov.tight_layout()

    if filename is not None:
        fig_mov.savefig(filename)


def _conv_imshow(image, filename=None, toplabel='', xlabel='', ylabel='', **kwargs):
    fig = plt.figure(figsize=(16, 12))
    ax  = fig.add_subplot(gs[0, 0])
    p   = ax.imshow(image, **kwargs)
    plt.colorbar(p)

    pt.addlabel(ax=ax, toplabel=toplabel, xlabel=xlabel, ylabel=ylabel)

    fig.tight_layout()

    plt.show()

    if filename is not None:
        fig.savefig(filename)

    return fig


def pearls(cam, filename=None, movie=False, verbose=False, debug=False, trunc=False):
    gs      = gridspec.GridSpec(1, 1)
    if movie:
        shutil.rmtree('movie_files')
        os.makedirs('movie_files')
        fig_mov = plt.figure(figsize=(8, 6))
        ax_mov  = fig_mov.add_subplot(gs[0, 0])
    # ======================================
    # Load file
    # ======================================
    if filename is None:
        data = E200.E200_load_data_gui()
    else:
        data = E200.E200_load_data(filename)
    # savefile = 'local.h5'
    # f = h5.File(savefile, 'r', driver='core', backing_store=False)
    # data = E200.Data(read_file = f)
    
    # ======================================
    # Get imgstr
    # ======================================
    imgstr = getattr(data.rdrill.data.raw.images, cam)
    uids = imgstr.UID
    uids = uids[uids > 1e5]
    if trunc:
        uids = uids[0:100]
    
    # ======================================
    # Load images
    # ======================================
    num_imgs = np.size(uids, 0)
    
    print('\nDone thresholding')
    
    # ======================================
    # Set up storage arrays
    # ======================================
    centroid        = np.empty((num_imgs, 2))
    for i, imgiter in enumerate(E200.E200_Image_Iter(imgstr, uids)):
        img = imgiter.images[0]
        if i % 10 == 0:
            sys.stdout.write('\rOn image number: {}'.format(i))
    
        if verbose:
            _conv_imshow(img, toplabel='Image', xlabel='X (px)', ylabel='Y (px)', filename='Threshold.png')
    
        # ======================================
        # Threshold images by half of the max
        # after median filter
        # ======================================
        thresh = skfilt.threshold_isodata(img)
        img_thresh = img > thresh
    
        if verbose:
            _conv_imshow(img_thresh, toplabel='Labeled Image', xlabel='Px', ylabel='Px', filename='Labels.png')
    
        if verbose:
            _conv_imshow(img_thresh, toplabel='Labeled Image', xlabel='Px', ylabel='Px', filename='Labels.png')
    
        img_thresh = skmorph.binary_erosion(img_thresh)
    
        skmorph.remove_small_objects(img_thresh, 50, in_place=True)
    
        if verbose:
            _conv_imshow(img_thresh, toplabel='Labeled Image', xlabel='Px', ylabel='Px', filename='Labels.png')
    
        # ======================================
        # Label each region
        # ======================================
        labels = skmeas.label(img_thresh, connectivity=1, background=0) + 1
    
        if verbose:
            _conv_imshow(labels, toplabel='Labeled Image', xlabel='Px', ylabel='Px', filename='Labels.png')
    
        # ======================================
        # Measure each region
        # ======================================
        props = skmeas.regionprops(labels, img)
    
        max_area = 0
        for j, prop in enumerate(props):
            if prop.euler_number < 0:
                filled = prop.filled_image
                filled = skmorph.binary_erosion(filled, selem=rad10)
                
                inserted = np.zeros(labels.shape, dtype=int)
                bbox = prop.bbox
                inserted[bbox[0]:bbox[0]+filled.shape[0], bbox[1]:bbox[1]+filled.shape[1]] = filled
    
                inserted_prop = skmeas.regionprops(inserted)[0]
                if inserted_prop.area > max_area:
                    max_area = inserted_prop.area
                    centroid[i, :] = inserted_prop.centroid
                
                # if verbose:
                #     fig = _conv_imshow(ana_img, toplabel='', xlabel='Px', ylabel='Px', filename='Labels.png')
                #     fig = _conv_imshow(canny_img, toplabel='', xlabel='Px', ylabel='Px', filename='Labels.png')
                #     plt.close(fig)
    
        if movie:
            movie_imshow(img, centroid=centroid[i, :], fig_mov=fig_mov, ax_mov=ax_mov, toplabel='Labeled Image', xlabel='Px', ylabel='Px', filename='movie_files/Labels_{:04d}.png'.format(i))
    
    if movie:
        fileinput = 'movie_files/Labels_%04d.png'
        savedir = '.'
        command = 'ffmpeg -y -framerate 10 -i {fileinput:} -vcodec h264 -r 30 -pix_fmt yuv420p {savedir:}/out_{dataset:}_{cam:}_pearls.mov'.format(fileinput=fileinput, savedir=savedir, dataset=data.loadname, cam=cam)
        subprocess.call(shlex.split(command))

    freq           = data.rdrill.data.raw.metadata.E200_state.EVNT_SYS1_1_BEAMRATE.dat[0]
    (f, xpow) = pt.fft(centroid[:, 1], freq=freq)
    (f, ypow) = pt.fft(centroid[:, 0], freq=freq)

    norm = np.max([xpow, ypow])
    xpow_norm = xpow/norm
    ypow_norm = ypow/norm
    
    fig = plt.figure()
    gs  = gridspec.GridSpec(1, 1)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.autoscale(tight=True)
    ax1.plot(f, xpow_norm, label='x')  # NOQA
    ax1.plot(f, ypow_norm, label='y')  # NOQA
    ax1.legend(loc=0)

    pt.addlabel(ax=ax1, toplabel='Fourier analysis of {dataset:}, {cam:}'.format(dataset=data.loadname, cam=cam), xlabel='Frequency (Hz)', ylabel='Power (norm.)')

    fig.tight_layout()

    fig.savefig('fourier_pearls_{dataset:}_{cam:}.eps'.format(dataset=data.loadname, cam=cam))
    fig.savefig('fourier_pearls_{dataset:}_{cam:}.png'.format(dataset=data.loadname, cam=cam))

    if debug:
        plt.ion()
        plt.show()
        ipdb.set_trace()
    else:
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=
            'Analyzes laser stability.')
    parser.add_argument('-V', action='version', version='%(prog)s v0.1')
    parser.add_argument('-v', '--verbose', action='store_true',
            help='Verbose mode.')
    parser.add_argument('-d', '--debug', action='store_true',
            help='Open debugger after running')
    parser.add_argument('-c', '--camera', required=True,
            help='Camera')
    parser.add_argument('-m', '--movie', action='store_true',
            help='Generate movie')
    parser.add_argument('-f', '--filename',
            help='Dataset filename')
    parser.add_argument('-t', '--test', action='store_true',
            help='Dataset filename')
    parser.add_argument('-trunc', action='store_true',
            help='Dataset filename')
    arg = parser.parse_args()

    if arg.test:
        import cProfile
        cProfile.run('pearls(True)', sort='tottime')

    pearls(cam=arg.camera, verbose=arg.verbose, movie=arg.movie, filename=arg.filename, debug=arg.debug, trunc=arg.trunc)
