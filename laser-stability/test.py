#!/usr/bin/env python3
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import sys
import skimage.filters as skfilt
import skimage.measure as skmeas
import skimage.morphology as skmorph
import skimage.segmentation as skseg
import pytools as pt
import E200
import numpy as _np
import ipdb

# verbose = True
verbose = False
reconstruct_radius = 2
gs  = gridspec.GridSpec(1, 1)


def std_distance(array):
    # ======================================
    # Return distance region is from mean,
    # normalized by std dev
    # ======================================
    return (array-_np.mean(array)) / _np.std(array)


def _conv_imshow(image, filename, toplabel, xlabel, ylabel):
    fig = plt.figure()
    gs  = gridspec.GridSpec(1, 1)
    ax  = fig.add_subplot(gs[0, 0])
    p   = ax.imshow(image, interpolation='none', cmap=mpl.cm.gray)
    plt.colorbar(p)

    pt.addlabel(ax=ax, toplabel=toplabel, xlabel=xlabel, ylabel=ylabel)

    fig.tight_layout()

    plt.show()

# ======================================
# Load file
# ======================================
# loadfile  = 'nas/nas-li20-pm00/E225/2015/20150601/E225_17682/E225_17698.mat'
#              nas/nas-li20-pm00/E225/2015/20150601/E225_17698/E225_17698.mat
data = E200.E200_load_data_gui()

# ======================================
# Get imgstr
# ======================================
cam  = 'IP2A'
imgstr = getattr(data.rdrill.data.raw.images, cam)
uids = imgstr.UID

# ======================================
# Load images
# ======================================
imgdat   = E200.E200_load_images(imgstr, uids)
imgs     = imgdat.imgs_subbed
num_imgs = _np.size(imgs, 0)

# ======================================
# Get threshold
# (half of the max after median filter)
# ======================================
thresh   = _np.empty(num_imgs)
imgs_max = 0

for i, img in enumerate(imgs):
    if i % 10 == 0:
        sys.stdout.write('\rOn image number: {}'.format(i))
    # disk      = skmorph.disk(1)
    # temp_img  = skfilt.median(img.astype('uint16'), selem=disk)
    img_max = _np.max(img)
    thresh[i] = img_max / 2
    imgs_max = _np.max((img_max, imgs_max))

avg_thresh = _np.mean(thresh)

print('\nDone thresholding')

# ======================================
# Set up storage arrays
# ======================================
centroid        = _np.empty((num_imgs, 2))
area            = _np.empty(num_imgs)
_regions        = _np.empty(num_imgs, dtype=object)
moments_central = _np.empty(num_imgs, dtype=object)
best_region    = _np.empty(num_imgs, dtype=object)
for i, img in enumerate(imgs):
    if i % 10 == 0:
        sys.stdout.write('\rOn image number: {}'.format(i))

    if verbose:
        _conv_imshow(img, toplabel='Image', xlabel='X (px)', ylabel='Y (px)', filename='Threshold.png')

    # ======================================
    # Threshold images by half of the max
    # after median filter
    # ======================================
    avg_thresh
    img_thresh = img > avg_thresh
    if verbose:
        _conv_imshow(img_thresh, toplabel='Threshold of Median', xlabel='X (px)', ylabel='Y (px)', filename='Threshold.png')

    # ======================================
    # Erode (eliminates noise, smooths
    # boundaries)
    # ======================================
    disk = skmorph.disk(reconstruct_radius)
    temp_img = skmorph.erosion(img_thresh, selem=disk)

    if _np.sum(temp_img) == 0:
        temp_img = img_thresh
        erode_later = True
    else:
        erode_later = False

    # ======================================
    # Dilate (returns to roughly original
    # size, further smooths)
    # ======================================
    disk    = skmorph.disk(reconstruct_radius)
    regions = skmorph.dilation(temp_img, selem=disk)

    if erode_later:
        disk = skmorph.disk(reconstruct_radius)
        regions = skmorph.erosion(regions, selem=disk)

    if verbose:
        _conv_imshow(regions, toplabel='Eroded and Dilated Image', xlabel='Px', ylabel='Px', filename='ErosionDilation.png')

    _regions[i] = regions

    # ======================================
    # Label each region
    # ======================================
    labels = skmeas.label(regions, connectivity=1, background=0) + 1

    if verbose:
        _conv_imshow(labels, toplabel='Labeled Image', xlabel='Px', ylabel='Px', filename='Labels.png')

    _labels = labels

    # ======================================
    # Measure each region
    # ======================================
    props = skmeas.regionprops(labels, img)

    num_regions  = _np.size(props)
    density      = _np.empty(num_regions)
    total_signal = _np.empty(num_regions)

    for j, prop in enumerate(props):
        density[j] = prop.mean_intensity
        total_signal[j] = prop.weighted_moments[0, 0]

    # ======================================
    # Merit is combination of density and
    # total signal
    # ======================================
    merit = std_distance(density) + std_distance(total_signal)

    # ======================================
    # Get measurements of best region
    # ======================================
    try:
        ind_select_region = _np.argmax(merit)
        best_region[i]    = props[ind_select_region]
        centroid[i, :]    = _np.array(best_region[i].weighted_centroid)
    except:
        best_region[i] = None
        centroid[i, :] = None

fig = plt.figure()
ax  = fig.add_subplot(gs[0, 0])
ax.plot(centroid[:, 0], label='x')
ax.plot(centroid[:, 1], label='y')

pt.addlabel(ax=ax, toplabel='Weighted Centroid', xlabel='Shot', ylabel='Position (px)')

ax.legend(loc=0)

ax.set_xlim([0, num_imgs])

fig.tight_layout()
# name = E200._numarray2str(data.rdrill.data.raw.metadata.param.save_back)
# ipdb.set_trace()
name = ''.join(data.rdrill.data.raw.metadata.param.save_name.view('S2').astype('str'))
author = 'E200 Python'
title  = 'Centroid Analysis: {}'.format(name)
text   = 'Centroid analysis of {}. Comment: {}'.format(cam, E200._numarray2str(data.rdrill.data.raw.metadata.param.comt_str))
file   = 'print.png'
link   = 'print.eps'

fig.savefig(file, dpi=50)
fig.savefig(link)

plt.show()

pt.facettools.print2elog(author=author, title=title, text=text, link=link, file=file)
