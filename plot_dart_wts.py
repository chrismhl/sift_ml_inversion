# Plot results from ML and compare them to true results at the 3 DART buoys
# Date: 6/6/2021
# Author: Christopher Liu

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from sklearn.metrics import explained_variance_score

import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
import matplotlib.gridspec as gridspec

# Was having problems installing and importing Geoclaw
# This is a quick workaround.
sys.path.insert(0,r'C:\Users\Chris\clawpack') 
from clawpack.geoclaw import dtopotools, topotools
from clawpack.clawutil.data import get_remote_file

# Define function to plot TS given a set of weights
def calc_ts(weights, buoy, us_eta, us_t):
    '''
    Calculates the time series from a linear combination of the unit sources as specified
    by the provided weights.
    Parameters
    ----------
    weights: npy array
        Weights corresponding to the unit sources in us_eta. Assumes weights
        are in the same order as the sources.
    buoy: str
       Buoy name used to access the correct unit sources from the us_eta dict
    us_eta: dict
        dict where the key is the buoy name and value is the pd dataframe containing
        the wave amplitude response from each individual unit source.
    us_t: dict
        dict where the key is the buoy name and value is the pd dataframe containing
        the time steps from each individual unit source.
    Returns
    ----------
    eta_tmp: npy array
       Calculated wave amplitude of the time series
    t_tmp: npy array
       Corresponding time steps to the amplitudes in eta_tmp
    '''
    eta_buoy = us_eta[buoy]
    t_buoy = us_t[buoy]
    
    eta_tmp = np.zeros(eta_buoy.shape[0])
    t_tmp = t_buoy.iloc[:,0].to_numpy()
    
    for n, wt in enumerate(weights):
        eta_tmp = eta_tmp + wt * eta_buoy.iloc[:,n].to_numpy()

    return eta_tmp, t_tmp

def plot_rectangular_slip(subfault,ax,cmin_slip,cmax_slip):
    '''
    Helper function for plotting the colored rectangles corresponding to slip on 
    a unit source
    Parameters
    ----------
    subfault: geoclaw.dtopotools.siftfault
        subfault from Geoclaw 
    ax: matplotlib.axes
         Axis to plot rectangles to
    cmin_slip: float
        Minimum value for the color bar, all slips below are transparent.
    cmax_slip: float
        Maximum value for the color bar
    '''
    x_corners = [subfault.corners[3][0],
                 subfault.corners[0][0],
                 subfault.corners[1][0],
                 subfault.corners[2][0],
                 subfault.corners[3][0]]

    y_corners = [subfault.corners[3][1],
                 subfault.corners[0][1],
                 subfault.corners[1][1],
                 subfault.corners[2][1],
                 subfault.corners[3][1]]
    
    slip = subfault.slip
    s = min(1, max(0, (slip-cmin_slip)/(cmax_slip-cmin_slip)))
    c = np.array(cmap_slip(s*.99))  # since 1 does not map properly with jet
    if slip <= cmin_slip:
        c[-1] = 0  # make transparent

    ax.fill(x_corners,y_corners,color=c,edgecolor='none')

def plot_subfaults(ax, slips, subfault, lat, title):
        '''
        Plot the unit sources and their corresponding slips with a fixed color map.
        Note: Consider moving the axes settings outside of the function
        Parameters
        ----------
        ax: matplotlib.axes
             Axis to plot rectangles to
        slips: dict
            Subfault names and corresponding slip in meters
        subfault: geoclaw.dtopotools.siftfault
            subfault from Geoclaw 
        lat: float
            latitude of the centroid of the slip calculated from the original fq run
        title: str
            title of the plot
        '''
        subfault.set_subfaults(slips)

        ax.plot(shore[:,0], shore[:,1], 'g')
        ax.axhline(y=lat, xmin=0, xmax=1, color ='red', ls='--', lw=1, alpha = 0.8)
        ax.set_aspect(1./np.cos(45*np.pi/180.))
        ax.axis([226,238,40.5,54.5]) # Plot ranges for axes
        ax.set_title('%s, Mw = %.2f' % (title, subfault.Mw()), fontsize=15)
        
        # Color bar range
        cmin_slip = 0.01  # min slip to plot, smaller values will be transparent
        cmax_slip = np.amax(np.array(list(slips.values())))

        for s in subfault.subfaults:
            c = s.corners
            c.append(c[0])
            c = np.array(c)
            ax.plot(c[:,0],c[:,1], 'k', linewidth=0.2)
            plot_rectangular_slip(s,ax,cmin_slip,cmax_slip)

        
        norm = mcolors.Normalize(vmin=cmin_slip,vmax=cmax_slip)
        cb1 = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap_slip),
                           ax=ax,shrink=0.4)
        cb1.set_label("Slip (m)",fontsize=12)
        cb1.ax.tick_params(labelsize=12)

if __name__ == "__main__":
    # read input file
    in_fpath = 'sift_ml_input.csv'
    if os.path.isfile(in_fpath):
         ml_input = pd.read_csv(in_fpath, dtype = {'unit_sources': str, 'dart': str,\
                                                 'lat_d': np.float64, 'long_d': np.float64,\
                                                 'extra_forecast': str, 'lat_f': np.float64,\
                                                 'long_f': np.float64})
    else:
        sys.exit("Error: Unit source file cannot be found.")
    
    cmap_slip = plt.cm.jet # set colormap for slip plot
    twin = 45 # time window in minutes
    outdir = 'conv_plots_31src_%s_300' % str(twin) #sources, time window, epochs
    savedir = os.path.join(outdir,'dart')
    dart = ml_input['dart'][ml_input.dart.notnull()].tolist()
    
    # load unit source dataframes
    dfd = 'unit_src_ts'
    eta_us = {}
    t_us = {}
    for name in dart:
        eta_us[name] = pd.read_csv(os.path.join(dfd,'eta_%s.csv' % name))
        t_us[name] = pd.read_csv(os.path.join(dfd,'t_%s.csv' % name))

    # Load fq data
    npyd = 'npy'
    eta = np.load(os.path.join(npyd,'fq_dart_eta.npy'))
    t = np.load(os.path.join(npyd,'fq_dart_time.npy'))

    # Load weights
    fq_wts_true = np.load(os.path.join(npyd,'fq_yong_inv_best.npy'))
    fq_wts_inv = [np.load(os.path.join(npyd,'fq_conv1d_wts_test_300.npy')),\
                  np.load(os.path.join(npyd,'fq_conv1d_wts_train_300.npy')),\
                  np.load(os.path.join(npyd,'fq_conv1d_wts_valid_300.npy'))]
    
    # Load indices
    inddir = 'indices'
    index = [np.loadtxt(os.path.join(inddir,'fq_dart_test_index.txt')).astype(int),\
                np.loadtxt(os.path.join(inddir,'fq_dart_train_index.txt')).astype(int),\
                np.loadtxt(os.path.join(inddir,'fq_dart_valid_index.txt')).astype(int)]
    runs = [np.loadtxt(os.path.join(inddir,'fq_dart_test_runs.txt')).astype(int),\
           np.loadtxt(os.path.join(inddir,'fq_dart_train_runs.txt')).astype(int),\
           np.loadtxt(os.path.join(inddir,'fq_dart_valid_runs.txt')).astype(int)]
    
    # Load lat data
    fq_lat = np.genfromtxt('fakequake_info.txt'\
                           ,delimiter=',')[:,2]
    
    # Load shoreline
    datadir = os.getcwd()
    filename = 'pacific_shorelines_east_4min.npy'
    shorelines_file = os.path.join(datadir, filename)
    try:
        shore = np.load(shorelines_file)
    except:
        url = 'http://depts.washington.edu/clawpack/geoclaw/topo/' + filename
        get_remote_file(url=url, output_dir=datadir, force=True, verbose=True)
        shore = load(shorelines_file)
    
    # Some more variables
    sfkey = ml_input['unit_sources'][ml_input.unit_sources.notnull()].tolist()
    sets = ['test', 'train','valid']
    
    # Plot
    SF = dtopotools.SiftFault()
    
    if not os.path.isdir(outdir):
            os.mkdir(outdir)
    if not os.path.isdir(savedir):
            os.mkdir(savedir)
    
    for s, name in enumerate(sets):
        plotdir = os.path.join(savedir,name)
        
        ind_tmp = index[s]
        runs_tmp = runs[s]
        wts_tmp = fq_wts_inv[s]
        
        # Extract the relevant data
        eta_fq = eta[ind_tmp, :, :]
        t_fq = t[ind_tmp, :, :]
        target = fq_wts_true[ind_tmp,:]

        if not os.path.isdir(plotdir):
            os.mkdir(plotdir)

        for n,r in enumerate(ind_tmp):
            #Create subplots/grid
            fig = plt.figure(constrained_layout=True, clear = True, figsize=(28,16))
            gs = fig.add_gridspec(3,4)
            axes = [fig.add_subplot(gs[0,0:2]),\
                    fig.add_subplot(gs[1,0:2]),\
                    fig.add_subplot(gs[2,0:2]),\
                    fig.add_subplot(gs[0:3,2]),\
                    fig.add_subplot(gs[0:3,3])
                    ]

            # Super title for plot
            fig.suptitle('Run # %s' % str(runs_tmp[n]), fontsize=20,)

            # Plot time series for each DART buoy.
            for b,buoy in enumerate(dart):
                # calculate ts and evs
                eta_i, t_i = calc_ts(wts_tmp[n,:], buoy, eta_us, t_us)
                eta_t, t_t = calc_ts(target[n,:], buoy, eta_us, t_us)
                evs = explained_variance_score(eta[r,b,:359],eta_i[:359])
            
                axes[b].plot(t_fq[n,b,:240]/60,eta_fq[n,b,:240], label = 'FQ Sol')
                axes[b].plot(t_i[:240]/60,eta_i[:240], label= 'ML Pred')
                axes[b].plot(t_t[:240]/60,eta_t[:240], label= 'SIFT Auto-Inversion')
                axes[b].axvline(x=twin/60, ymin=0, ymax=1, color ='red', ls='--', lw=1, alpha = 0.8)
                axes[b].set_title("Buoy: %s, EVS: %s" % (buoy, str(round(evs,2))))

                if b == 0:
                    axes[b].legend()
                elif b == 1:
                    axes[b].set_ylabel('Height (meters)')
                elif b == 2:
                    axes[b].set_xlabel('Time (Hours)')

            # Plot weights from ML prediction
            sift_slip = {}
            for m,k in enumerate(sfkey):
                sift_slip[k] = wts_tmp[n,m]

            plot_subfaults(axes[-2], sift_slip, SF, fq_lat[runs_tmp[n]], 'ML Pred')

            sift_slip = {s : 0 for s in sfkey}

            # Plot the true weights
            for m,k in enumerate(sfkey):
                sift_slip[k] = target[n,m]

            plot_subfaults(axes[-1], sift_slip, SF, fq_lat[runs_tmp[n]], 'LS Inv')

            # Save name
            fname = 'ml_inv_run%s.png' % str(runs_tmp[n]).zfill(4)

            plt.savefig(os.path.join(plotdir, fname))
            fig.clear()
            plt.close(fig)