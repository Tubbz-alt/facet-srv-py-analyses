#!/usr/bin/env python3
import ipdb
import numpy as np
import pytools as mt
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import stability as st

print('Here')
blobs   = st.run_analysis()
print('There')
camlist = ['AX_IMG', 'AX_IMG2']


gs  = gridspec.GridSpec(2, 1)
fig = plt.figure(figsize = (8, 12))

for i, (name, blob) in enumerate(zip(camlist, blobs)):
    ax1 = fig.add_subplot(gs[i, 0])
    
    num_samples = np.size(blob.centroid_avg[:, 0])
    interval    = 1
    xfft = np.fft.rfft(blob.centroid_avg[:, 0])
    yfft = np.fft.rfft(blob.centroid_avg[:, 1])

    factor = 1.0/(interval*num_samples)
    num_fft = np.size(xfft)
    f = factor * np.linspace(1, num_fft, num_fft)

    xpow = np.abs(xfft*np.conj(xfft))
    ypow = np.abs(yfft*np.conj(yfft))
    
    norm = np.max([xpow, ypow])
    px = ax1.plot(f, xpow/norm, label='x')
    py = ax1.plot(f, ypow/norm, label='y')
    
    ax1.legend(loc=0)
    
    mt.addlabel(ax=ax1, toplabel='Fourier analysis of {}'.format(name), xlabel='Frequency (Hz)', ylabel='Power (norm.)')

fig.tight_layout()
plt.ioff()
plt.show()

fig.savefig('formike.png')
