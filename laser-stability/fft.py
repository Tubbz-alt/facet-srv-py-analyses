#!/usr/bin/env python3
import ipdb  # NOQA
import numpy as np
import pytools as pt
import pytools.qt as ptqt
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
# import stability as st


def fft(blobs, camlist, fill_missing=False, freq=None):
    # ======================================
    # Get frequency
    # ======================================
    if freq is None:
        freq = ptqt.getDouble(title='Fourier Analysis', text='Frequency samples taken at:', min=0, decimals=2, value=1.0)
        freq = freq.input
    
    fig = plt.figure(figsize=(16, 6))
    gs = gridspec.GridSpec(1, 2)
    
    for i, (name, blob) in enumerate(zip(camlist, blobs)):
        ax1 = fig.add_subplot(gs[0, i])
        # ax1 = fig.add_subplot(gs[i, 0])

        if fill_missing:
            (t_x, x_filled) = pt.fill_missing_timestamps(blob.timestamps, blob.centroid[:, 0])
            (t_x, y_filled) = pt.fill_missing_timestamps(blob.timestamps, blob.centroid[:, 1])
        else:
            x_filled = blob.centroid[:, 0]
            y_filled = blob.centroid[:, 1]
        
        num_samples = np.size(x_filled)
        xfft = np.fft.rfft(x_filled)
        yfft = np.fft.rfft(y_filled)
    
        factor = freq/num_samples
        num_fft = np.size(xfft)
        f = factor * np.linspace(1, num_fft, num_fft)
    
        xpow = np.abs(xfft*np.conj(xfft))
        ypow = np.abs(yfft*np.conj(yfft))

        # ======================================
        # No DC term
        # ======================================
        xpow = xpow[1:]
        ypow = ypow[1:]
        f = f[1:]

        norm = np.max([xpow, ypow])
        px = ax1.plot(f, xpow/norm, label='x')  # NOQA
        py = ax1.plot(f, ypow/norm, label='y')  # NOQA
        
        ax1.legend(loc=0)
        
        pt.addlabel(ax=ax1, toplabel='Fourier analysis of {}'.format(name), xlabel='Frequency (Hz)', ylabel='Power (norm.)')
    
    mainfigtitle = 'Fourier Analysis'
    fig.suptitle(mainfigtitle, fontsize=22)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    return fig
