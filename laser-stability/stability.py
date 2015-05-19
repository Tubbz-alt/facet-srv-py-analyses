#!/usr/bin/env python3
from mpl_toolkits import mplot3d as m3d
import E200
import argparse
import h5py as h5
import ipdb
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import mytools as mt
# import mytools.imageprocess as mtimg
import classes as mtimg
import numpy as np
import os
import shlex
import subprocess
import sys
import tempfile


def run_analysis(save=False, check=False, debug=False, verbose=False, movie=False, pdf=None):
    # ======================================
    # Prep save folder
    # ======================================
    savedir = 'output'
    if not os.path.isdir(savedir):
        os.makedirs(savedir)

    # ======================================
    # Load data
    # ======================================
    savefile = os.path.join(os.getcwd(), 'local.h5')
    data = E200.E200_load_data('nas/nas-li20-pm00/E217/2015/20150504/E217_16808/E217_16808.mat', savefile=savefile)
    # f = h5.File(savefile, 'r', driver='core', backing_store=False)
    # data = E200.Data(read_file = f)
    
    # ======================================
    # Cameras to process
    # ======================================
    camlist      = ['AX_IMG', 'AX_IMG2']
    radii        = [2, 1]
    calibrations = [10e-6, 17e-6]

    blobs = np.empty(2, dtype=object)

    for i, (cam, radius, cal) in enumerate(zip(camlist, radii, calibrations)):
        imgstr = getattr(data.rdrill.data.raw.images, cam)
        blob = mtimg.BlobAnalysis(imgstr, imgname=cam, cal=cal, reconstruct_radius=1, check=check, debug=debug, verbose=verbose, movie=movie, save=save)
        if save or check or (pdf is not None):
            fig = blob.camera_figure(save=save)
            if pdf is not None:
                pdf.savefig(fig)
            if check:
                plt.show()
        blobs[i] = blob

    # ======================================
    # Process centroids into array of coords
    # correlated for 3d plotting
    # ======================================
    z = np.array((0, 1.5))
    coords = np.empty([0, 100, 3])
    for i, blob in enumerate(blobs):
        z = i*1.5 * np.ones((np.size(blob.centroid, 0), 1))
        # ipdb.set_trace()
        temp_cent = blob.centroid
        centered = temp_cent - np.mean(temp_cent, axis=0)
        coord = np.append(centered, z, axis=1)
        coords = np.append(coords, [coord], axis=0)

    # ======================================
    # Plot relative 3D trajectories
    # ======================================
    fig = plt.figure(figsize=(16, 6))
    gs = gridspec.GridSpec(1, 2)
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    ax2 = fig.add_subplot(gs[0, 1], projection='3d')
    # maxwidth = np.max(blobs[0].sigma_x * blobs[0].sigma_y)
    # widths = blobs[0].sigma_x[i]*blobs[0].sigma_y[i] / maxwidth
    for i, coord in enumerate(coords.swapaxes(0, 1)):
        ax1.plot(coord[:, 2], coord[:, 0]*1e6, coord[:, 1]*1e6)
        ax2.plot(coord[:, 2], coord[:, 0]*1e6, coord[:, 1]*1e6)
    mt.addlabel(ax=ax1, toplabel='Centroid Trajectory', xlabel='z [m]', ylabel='x [$\mu$m]', zlabel='y [$\mu$m]')
    mt.addlabel(ax=ax2, toplabel='Centroid Trajectory', xlabel='z [m]', ylabel='x [$\mu$m]', zlabel='y [$\mu$m]')
    fig.tight_layout()

    # ======================================
    # Make a movie
    # ======================================
    if movie:
        with tempfile.TemporaryDirectory() as tempdir:
            for ii in range(0, 360, 1):
                sys.stdout.write('\rFrame: {}'.format(ii))
                ax1.view_init(elev=45., azim=ii-60)
                ax2.view_init(elev=0., azim=ii-60)
                fig.savefig(os.path.join(tempdir, 'movie_{:03d}.tif'.format(ii)))

            fileinput = os.path.join(tempdir, 'movie_%03d.tif')
            command = 'ffmpeg -y -framerate 30 -i {fileinput:} -vcodec h264 -r 30 -pix_fmt yuv420p {savedir:}/out.mov'.format(fileinput=fileinput, savedir=savedir)
            subprocess.call(shlex.split(command))

    # ======================================
    # Calculate angles
    # ======================================
    dx = coords[0, :, 0] - coords[1, :, 0]
    dy = coords[0, :, 1] - coords[1, :, 1]
    ds = np.sqrt(dx**2 + dy**2)
    theta = np.arctan(ds/z.flat)
    theta_urad = theta * 1e6
    phi = np.arctan(dy/dx)

    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(theta_urad, '-o')
    mt.addlabel(ax=ax1, toplabel='Coordinate: $\\theta$', xlabel='Shot', ylabel='Angle Deviation from Average [$\mu$rad]')

    ax2 = fig.add_subplot(gs[1, 0])
    mt.hist(theta_urad, bins=15, ax=ax2)
    mt.addlabel(ax=ax2, toplabel='Coordinate: $\\theta$', xlabel='Angle Deviation from Average [$\mu$rad]')

    ax3 = fig.add_subplot(gs[0, 1])
    ax3.plot(phi, '-o')
    mt.addlabel(ax=ax3, toplabel='Coordinate: $\phi$', xlabel = 'Shot', ylabel='Direction of Deviation from Average [rad]')

    ax4 = fig.add_subplot(gs[1, 1])
    mt.hist(phi, bins=15, ax=ax4)
    mt.addlabel(ax=ax4, toplabel='Coordinate: $\phi$', xlabel='Direction of Deviation from Average [rad]')

    mainfigtitle = 'Pointing Stability'
    fig.suptitle(mainfigtitle, fontsize=22)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save or (pdf is not None):
        fig.savefig(os.path.join(savedir, 'PointingStability.eps'))
        pdf.savefig(fig)

    if debug:
        plt.ion()
        plt.show()
        ipdb.set_trace()
    else:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=
            'Creates a tunnel primarily for Git.')
    parser.add_argument('-V', action='version', version='%(prog)s v0.1')
    parser.add_argument('-v', '--verbose', action='store_true',
            help='Verbose mode.')
    parser.add_argument('-s', '--save', action='store_true',
            help='Save movie files')
    parser.add_argument('-c', '--check', action='store_true',
            help='View analysis')
    parser.add_argument('-d', '--debug', action='store_true',
            help='Open debugger after running')
    parser.add_argument('-m', '--movie', action='store_true',
            help='Generate movie')
    parser.add_argument('-p', '--pdf', action='store_true',
            help='Generate pdf')
    arg = parser.parse_args()

    if arg.pdf:
        with PdfPages('output.pdf') as pdf:
            run_analysis(save=arg.save, check=arg.check, debug=arg.debug, verbose=arg.verbose, movie=arg.movie, pdf=pdf)
    else:
        run_analysis(save=arg.save, check=arg.check, debug=arg.debug, verbose=arg.verbose, movie=arg.movie)