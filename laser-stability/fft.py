#!/usr/bin/env python3
import ipdb
import numpy as np
import pytools as pt
import pytools.qt as ptqt
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
# import stability as st


def fft(blobs, camlist):
    # ipdb.set_trace()
    freq = ptqt.getDouble(title='Fourier Analysis', text='Frequency samples taken at:', min=0, decimals=2, value=1.0)
    # blobs   = st.run_analysis()
    # camlist = ['AX_IMG', 'AX_IMG2']
    
    fig = plt.figure(figsize=(16, 6))
    gs = gridspec.GridSpec(1, 2)
    
    for i, (name, blob) in enumerate(zip(camlist, blobs)):
        ax1 = fig.add_subplot(gs[0, i])
        # ax1 = fig.add_subplot(gs[i, 0])
        
        num_samples = np.size(blob.centroid_avg[:, 0])
        xfft = np.fft.rfft(blob.centroid_avg[:, 0])
        yfft = np.fft.rfft(blob.centroid_avg[:, 1])
    
        factor = freq.input/num_samples
        num_fft = np.size(xfft)
        f = factor * np.linspace(1, num_fft, num_fft)
    
        xpow = np.abs(xfft*np.conj(xfft))
        ypow = np.abs(yfft*np.conj(yfft))
        
        norm = np.max([xpow, ypow])
        px = ax1.plot(f, xpow/norm, label='x')
        py = ax1.plot(f, ypow/norm, label='y')
        
        ax1.legend(loc=0)
        
        pt.addlabel(ax=ax1, toplabel='Fourier analysis of {}'.format(name), xlabel='Frequency (Hz)', ylabel='Power (norm.)')
    
    mainfigtitle = 'Fourier Analysis'
    fig.suptitle(mainfigtitle, fontsize=22)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    return fig
