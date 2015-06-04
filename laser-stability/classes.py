import E200
import ipdb
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pytools as mt
import numpy as _np
import os
import shlex
import skimage.filters as skfilt
import skimage.measure as skmeas
import skimage.morphology as skmorph
import skimage.segmentation as skseg
import subprocess
import sys

__all__ = ['BlobAnalysis']


# ======================================
# Data storage class for analysis
# ======================================
class BlobAnalysis(object):
    # ======================================
    # Initialize class
    # ======================================
    def __init__(self, imgstr, imgname, cal, reconstruct_radius=2, uids=None, verbose=False, check=False, save=False, debug=False, movie=False, beamtype=None):
        self.imgstr             = imgstr
        self.imgname            = imgname
        self.reconstruct_radius = reconstruct_radius
        self.uids               = uids
        self.verbose            = verbose
        self.check              = check
        self.save               = save
        self.debug              = debug
        self.movie              = movie
        self.cal                = cal
        self.beamtype           = beamtype

        if check or save or movie:
            self.gs  = gridspec.GridSpec(1, 1)
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(self.gs[0, 0])
            self.fig.tight_layout()

        self._preprocess_images()
        self._process_images(None)

    # ======================================
    # Main Program
    # ======================================
    def _process_images(self, lookup):
        if lookup is None:
            print('Processing images...')
            centroid          = _np.empty((self.num_imgs, 2))
            area              = _np.empty(self.num_imgs)
            # self._regions     = _np.empty(self.num_imgs, dtype=object)
            moments_central   = _np.empty(self.num_imgs, dtype=object)
            # self._best_region = _np.empty(self.num_imgs, dtype=object)
            self._timestamps = _np.empty(self.num_imgs)

            # ======================================
            # Iterate over images
            # ======================================
            for i, imgiter in enumerate(E200.E200_Image_Iter(self.imgstr, uids=self.uids)):
                img = imgiter.images[0]
                self._timestamps[i] = imgiter.timestamps[0]
                if i % 10 == 0:
                    sys.stdout.write('\rOn image number: {}'.format(i))

                # ======================================
                # Threshold images by half of the max
                # after median filter
                # ======================================
                if self.verbose:
                    avg_thresh = 100
                else:
                    avg_thresh = self.avg_thresh
                img_thresh = img > avg_thresh
                if self.verbose:
                    self._conv_imshow(img_thresh, toplabel='Threshold of Median', xlabel='X (px)', ylabel='Y (px)', filename='Threshold.png')
                    plt.show()

                # ======================================
                # Erode (eliminates noise, smooths
                # boundaries)
                # ======================================
                disk = skmorph.disk(self.reconstruct_radius)
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
                disk    = skmorph.disk(self.reconstruct_radius)
                regions = skmorph.dilation(temp_img, selem=disk)

                if erode_later:
                    disk = skmorph.disk(self.reconstruct_radius)
                    regions = skmorph.erosion(regions, selem=disk)

                if self.verbose:
                    self._conv_imshow(regions, toplabel='Eroded and Dilated Image', xlabel='Px', ylabel='Px', filename='ErosionDilation.png')
                    plt.show()

                # self._regions[i] = regions

                # ======================================
                # Label each region
                # ======================================
                labels = skmeas.label(regions, connectivity=1, background=0) + 1

                if self.verbose:
                    self._conv_imshow(labels, toplabel='Labeled Image', xlabel='Px', ylabel='Px', filename='Labels.png')
                    plt.show()

                self._labels = labels

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
                merit = self.std_distance(density) + self.std_distance(total_signal)
    
                # ======================================
                # Get measurements of best region
                # ======================================
                ind_select_region      = _np.argmax(merit)
                # self._best_region[i] = props[ind_select_region]
                best_region            = props[ind_select_region]
                centroid[i, :]         = _np.array(best_region.weighted_centroid)
                area[i]                = best_region.area
                moments_central[i]     = best_region.moments_central
                # print('Centroid: {}'.format(centroid[i, :]))
    
                # ======================================
                # Get border of best region
                # ======================================
                best_region_label = (best_region.label == labels)
                border = skseg.find_boundaries(best_region_label.astype('int'), mode='outer')
                border = _np.ma.masked_equal(border, 0)
    
                # ======================================
                # Clear create updated figure
                # ======================================
                if self.check or self.movie or self.save:
                    self.ax.cla()
    
                    middle_x = centroid[i, 1]
                    middle_y = img.shape[0] - centroid[i, 0] - 1

                    # rot_amt    = 0
                    # plt_img    = _np.rot90(img, rot_amt)
                    # plt_border = _np.rot90(border, rot_amt)
                    plt_img    = _np.flipud(img)
                    plt_border = _np.flipud(border)

                    # x, y = mt.pcolor_axes(plt_img, px_to_units=px_to_units)
                    # x, y = mt.pcolor_axes(plt_img)
                    x, y = mt.NonUniformImage_axes(plt_img)
                    x = (x-middle_x) * self.cal * 1e6
                    y = (y-middle_y) * self.cal * 1e6
                    
                    mt.NonUniformImage(x, y, plt_img, ax=self.ax, cmap=mpl.cm.cool, vmax=self.imgs_max)
                    mt.NonUniformImage(x, y, plt_border, ax=self.ax, cmap=mpl.cm.gray_r, alpha=0.6)
                    self.ax.plot(0, 0, 'wo', scalex=False, scaley=False)
                    self.ax.set_aspect(1)
                    # self.ax.plot(middle_x, middle_y, 'wo', scalex=False, scaley=False)

                    # self.ax.imshow(img, interpolation='none', cmap=mpl.cm.cool)
                    # self.ax.imshow(border, interpolation='none', cmap=mpl.cm.gray_r, alpha=0.6)
                    # self.ax.plot(centroid[i, 1], centroid[i, 0], 'wo', scalex=False, scaley=False)

                    mt.addlabel(ax=self.ax, toplabel='Selected Region, Shot: {}'.format(i), xlabel='X ($\mu$m)', ylabel='Y ($\mu$m)')

                    self.fig.tight_layout()
    
                # ======================================
                # Clear create updated figure
                # ======================================
                if self.verbose:
                    vfig = plt.figure()
                    vax  = vfig.add_subplot(self.gs[0, 0])
                    vp1  = vax.imshow(img, interpolation='none', cmap=mpl.cm.cool, vmax=self.imgs_max)
                    vax.imshow(border, interpolation='none', cmap=mpl.cm.gray_r, alpha=0.6)
                    vax.plot(centroid[i, 1], centroid[i, 0], 'wo', scalex=False, scaley=False)

                    plt.colorbar(vp1)
                    mt.addlabel(ax=vax, toplabel='Selected Region', xlabel='X (px)', ylabel='Y (px)')

                    vfig.tight_layout()

                    filename = 'Measured.png'
                    vfig.savefig(filename, bbox_inches='tight')
                    plt.close(vfig)

                    self.verbose = False

                # ======================================
                # Draw updated figure
                # ======================================
                if self.check:
                    plt.draw()
                    plt.pause(0.0001)
                    if self.debug:
                        ipdb.set_trace()
    
                if self.movie or self.save:
                    filename = 'movie_files/movie_{}_{:03d}.tif'.format(self.imgname, i)

                    try:
                        self.fig.savefig(filename)
                    except FileNotFoundError:
                        os.makedirs(os.path.dirname(filename))
                        self.fig.savefig(filename)
                        
            if self.check or self.movie:
                plt.close(self.fig)

            # ======================================
            # Post analysis
            # ======================================
            centroid_avg = _np.empty(centroid.shape)
            centroid_avg[:, 0] = centroid[:, 0]-_np.mean(centroid[:, 0])
            centroid_avg[:, 1] = centroid[:, 1]-_np.mean(centroid[:, 1])
            sigma_x      = _np.empty(self.num_imgs)
            sigma_y      = _np.empty(self.num_imgs)
            for i, moment_central in enumerate(moments_central):
                sigma_x[i] = _np.sqrt(moment_central[2, 0])
                sigma_y[i] = _np.sqrt(moment_central[0, 2])

            self._area            = area * self.cal**2
            self._centroid        = centroid * self.cal
            self._centroid_avg    = centroid_avg * self.cal
            self._moments_central = moments_central
            self._sigma_x         = sigma_x * self.cal
            self._sigma_y         = sigma_y * self.cal

            print('')
            print('Centroid jitter (std. dev.)-\n\tX: {:0.2f}\t\tY: {:0.2f}'.format(_np.std(centroid_avg[:, 0]), _np.std(centroid_avg[:, 1])))
            print('Avg. Std. Dev.-\n\tX: {:0.2f}\tY: {:0.2f}'.format(_np.mean(sigma_x), _np.mean(sigma_y)))
            
            if self.movie:
                savedir = 'output'
                if not os.path.isdir(savedir):
                    os.makedirs(savedir)

                command = 'ffmpeg -y -framerate 10 -i movie_files/movie_{val:}_%03d.tif -vcodec h264 -r 30 -pix_fmt yuv420p {savedir:}/out_{val:}.mov'.format(val=self.imgname, savedir=savedir)
                subprocess.call(shlex.split(command))
    
            if self.debug:
                ipdb.set_trace()

            # plt.close('all')

    def camera_figure(self, save=False, savefig=None, dataset=''):
        # ======================================
        # Set up plots
        # ======================================
        gs  = gridspec.GridSpec(2, 2)
        fig = plt.figure(figsize=(16, 12))
        
        # ======================================
        # Plot centroid, area
        # ======================================
        ax1   = fig.add_subplot(gs[0, 0])
        size  = self.area / _np.max(self.area) * 80
        color = _np.linspace(1, self.num_imgs, self.num_imgs)
        ax1_p = ax1.scatter(self.centroid_avg[:, 0]*1e6, self.centroid_avg[:, 1]*1e6, c=color, marker='o', s=size, cmap=plt.get_cmap('Greys'))
        
        cb    = plt.colorbar(ax1_p)
        xstd = _np.std(self.centroid_avg[:, 0]*1e6)
        ystd = _np.std(self.centroid_avg[:, 1]*1e6)
        mt.addlabel(toplabel='Beam Centroid Evolution\n(Circle Area Proportional to Region Area)', xlabel='X position ($\mu$m), Std. Dev. {:0.02f}'.format(xstd), ylabel='Y position ($\mu$m), Std. Dev. {:0.02f}'.format(ystd), ax=ax1, cb=cb, clabel='Shot number')
        
        # fig.colorbar(p)
        # ======================================
        # Area plot
        # ======================================
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(self.area * 1e12, '-o')
        avgarea = _np.mean(self.area*1e12)
        stdarea = _np.std(self.area*1e12)
        mt.addlabel(toplabel='Region Above Threshold: Area', ylabel='Area ($\mu$m$^2$), Avg. {:0.02f}, Std. {:0.02f}'.format(avgarea, stdarea), xlabel='Shot', ax=ax2)
        
        # ======================================
        # Sigma_x
        # ======================================
        ax3    = fig.add_subplot(gs[1, 0])
        ax3.plot(self.sigma_x * 1e6, '-o', label='$\sigma_x$')
        ax3.plot(self.sigma_y * 1e6, '-o', label='$\sigma_y$')
        ax3.legend(loc=0)
        mt.addlabel(toplabel='Region Above Threshold: Std. Dev. of Area', ylabel='Std. Dev. ($\mu$m)', xlabel='Shot', ax=ax3)

        # ======================================
        # Sigma_y
        # ======================================
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(self.sigma_x/self.sigma_y, '-o')
        mt.addlabel(toplabel=r'Region Above Threshold: Std. Dev. Ratio', ylabel='$\sigma_x/\sigma_y$', xlabel='Shot', ax=ax4)
        
        mainfigtitle = r'{} {} Stability Analysis'.format(dataset, self.imgname)
        fig.suptitle(mainfigtitle, fontsize=22)
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        if save:
            if savefig is None:
                savefig = 'output/{}.eps'.format(mainfigtitle)

            try:
                fig.savefig(savefig)
            except FileNotFoundError:
                os.makedirs(os.path.dirname(savefig))
                fig.savefig(savefig)

        return fig

    # ======================================
    # Helper functions
    # ======================================
    def _preprocess_images(self):
        print('Preprocessing images...')
        # ======================================
        # Preprocess images
        # ======================================
        imgs_max = 0
        thresh = _np.empty(self.num_imgs)
        for i, imgiter in enumerate(E200.E200_Image_Iter(self.imgstr, uids=self.uids)):
            if i % 10 == 0:
                sys.stdout.write('\rOn image number: {}'.format(i))
            # sys.stdout.write('\rOn image number: {}'.format(i))
            img = imgiter.images[0]
            # ======================================
            # Get threshold
            # (half of the max after median filter)
            # ======================================
            disk      = skmorph.disk(1)
            temp_img  = skfilt.median(img.astype('uint16'), selem=disk)
            img_max = _np.max(temp_img)
            thresh[i] = img_max / 2
    
            imgs_max = _np.max((img_max, imgs_max))
            # ipdb.set_trace()
    
        self._thresh = thresh
        self._imgs_max = imgs_max
        self._avg_thresh = _np.mean(thresh)
        # logger.info('Average threshold is: {}'.format(avg_thresh), level=level)
        print('')

    def _conv_imshow(self, image, filename, toplabel, xlabel, ylabel):
        fig = plt.figure()
        gs  = gridspec.GridSpec(1, 1)
        ax  = fig.add_subplot(gs[0, 0])
        p   = ax.imshow(image, interpolation='none', cmap=mpl.cm.gray)
        plt.colorbar(p)

        mt.addlabel(ax=ax, toplabel=toplabel, xlabel=xlabel, ylabel=ylabel)
    
        fig.tight_layout()
    
        fig.savefig(filename, bbox_inches='tight')
        plt.close(fig)

    def std_distance(self, array):
        # ======================================
        # Return distance region is from mean,
        # normalized by std dev
        # ======================================
        return (array-_np.mean(array)) / _np.std(array)

    def _reset_calcs(self):
        self._area            = None
        self._avg_thresh      = None
        # self._best_region     = None
        self._centroid        = None
        self._centroid_avg    = None
        self._imgs_max        = None
        self._labels          = None
        self._moments_central = None
        self._num_imgs        = None
        # self._regions         = None
        self._thresh          = None

    # ======================================
    # Input data
    # ======================================
    @property
    def cal(self):
        return self._cal

    @cal.setter
    def cal(self, value):
        self._reset_calcs()
        self._cal = value

    @property
    def imgs(self):
        if self._imgs is None:
            # ======================================
            # Load images
            # ======================================
            imgs      = E200.E200_load_images(self.imgstr, self.uids)
            self._imgs = imgs.images
            self._num_imgs = _np.size(self._imgs, 0)

        return self._imgs

    @property
    def imgstr(self):
        return self._imgstr

    @imgstr.setter
    def imgstr(self, value):
        self._reset_calcs()
        self._imgstr = value

    @property
    def reconstruct_radius(self):
        return self._reconstruct_radius

    @reconstruct_radius.setter
    def reconstruct_radius(self, value):
        self._reset_calcs()
        self._reconstruct_radius = value

    @property
    def uids(self):
        return self._uids

    @uids.setter
    def uids(self, value):
        if value is None:
            self._uids = self.imgstr.UID
        else:
            self._uids = value
        self._imgs = None

    # ======================================
    # Calculated Properties
    # ======================================
    @property
    def area(self):
        self._process_images(self._area)
        return self._area

    @property
    def avg_thresh(self):
        if self._avg_thresh is None:
            self._preprocess_images()

        return self._avg_thresh

    @property
    def centroid(self):
        self._process_images(self._centroid)
        return self._centroid

    @property
    def centroid_avg(self):
        self._process_images(self._centroid_avg)
        return self._centroid_avg

    @property
    def imgs_max(self):
        if self._imgs_max is None:
            self._preprocess_images()

        return self._imgs_max

    @property
    def moments_central(self):
        self._process_images(self._moments_central)
        return self._moments_central

    @property
    def num_imgs(self):
        if self._num_imgs is None:
            self._num_imgs = _np.size(self.uids, 0)
            
        return self._num_imgs
    
    @property
    def sigma_x(self):
        self._process_images(self._sigma_x)
        return self._sigma_x

    @property
    def sigma_y(self):
        self._process_images(self._sigma_y)
        return self._sigma_y

    @property
    def thresh(self):
        self._process_images(self._thresh)
        return self._thresh

    # ======================================
    # Intermediate Images
    # ======================================
    # @property
    # def regions(self):
    #     self._process_images(self._regions)
    #     return self._regions
    
    # @property
    # def best_region(self):
    #     self._process_images(self._best_region)
    #     return self._best_region

    @property
    def labels(self):
        return self._labels
