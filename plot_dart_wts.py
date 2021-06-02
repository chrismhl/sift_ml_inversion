# Plot results from ML and compare them to true results
# Add flexibility to plot test, train, and valid

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from scipy.optimize import lsq_linear
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
    eta_buoy = us_eta[buoy]
    t_buoy = us_t[buoy]
    
    eta_tmp = np.zeros(eta_buoy.shape[0])
    t_tmp = t_buoy.iloc[:,0].to_numpy()
    
    for n, wt in enumerate(weights):
        eta_tmp = eta_tmp + wt * eta_buoy.iloc[:,n].to_numpy()

    return eta_tmp, t_tmp

# Helper fucntion for plotting the rectangles
def plot_rectangular_slip(subfault,ax,cmin_slip,cmax_slip):
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

# Plot subfaults and their weights
# Consider moving the axes settings outside of the function?
def plot_subfaults(ax, slips, subfault, lat, title):
        subfault.set_subfaults(slips)

        ax.plot(shore[:,0], shore[:,1], 'g')
        ax.axhline(y=lat, xmin=0, xmax=1, color ='red', ls='--', lw=1, alpha = 0.8)
        ax.set_aspect(1./np.cos(45*np.pi/180.))
        ax.axis([226,238,40.5,54.5])
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
    cmap_slip = plt.cm.jet # set colormap for slip plot
    twin = 45
    n_sources = 31
    plotdir = r'conv_plots_%ssrc_%s_300\dart' % (str(n_sources),str(twin))
    dart = ['46404', '46407', '46419']
    
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
    fq_wts_inv = np.load(os.path.join(npyd,'fq_conv_wts_300.npy'))
    
    # Load indices, not all used
    inddir = 'indices'
    train_runs = np.loadtxt(os.path.join(inddir,'fq_dart_train_runs.txt')).astype(int)
    test_runs = np.loadtxt(os.path.join(inddir,'fq_dart_test_runs.txt')).astype(int)
    train_ind = np.loadtxt(os.path.join(inddir,'fq_dart_train_index.txt')).astype(int)
    test_ind = np.loadtxt(os.path.join(inddir,'fq_dart_test_index.txt')).astype(int)
    
    # Extract the relevant runs
    eta_ts = eta[test_ind, :, :]
    t_ts = t[test_ind, :, :]
    target_ts = fq_wts_true[test_ind,:]
    
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
    buoys = ['46404', '46407', '46419']
    sfkey = list(eta_us['46404'].columns)
    
    # Plot
    SF = dtopotools.SiftFault()

    if not os.path.isdir(plotdir):
        os.mkdir(plotdir)

    for n,r in enumerate(test_ind):
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
        fig.suptitle('Run # %s' % str(test_runs[n]), fontsize=20,)
        
        # Plot time series for each DART buoy.
        for b,buoy in enumerate(buoys):
            eta_i, t_i = calc_ts(fq_wts_inv[n,:], buoy, eta_us, t_us)
            eta_t, t_t = calc_ts(target_ts[n,:], buoy, eta_us, t_us)
            evs = explained_variance_score(eta[r,b,:359],eta_i[:359])

            axes[b].plot(t[r,b,:240]/60,eta[r,b,:240], label = 'FQ Sol')
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
            sift_slip[k] = fq_wts_inv[n,m]

        plot_subfaults(axes[-2], sift_slip, SF, fq_lat[test_runs[n]], 'ML Pred')

        sift_slip = {s : 0 for s in sfkey}
        
        # Plot the true weights
        for m,k in enumerate(sfkey):
            sift_slip[k] = target_ts[n,m]

        plot_subfaults(axes[-1], sift_slip, SF, fq_lat[test_runs[n]], 'LS Inv')
        
        # Save name
        fname = 'ml_inv_run%s.png' % str(test_runs[n]).zfill(4)
        
        plt.savefig(os.path.join(plotdir, fname))
        fig.clear()
        plt.close(fig)