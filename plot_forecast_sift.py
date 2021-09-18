# Plot comparison results at the additional forecast points
# Date: 6/6/2021
# Author: Christopher Liu

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from sklearn.metrics import explained_variance_score

# Removing the plot for the SJdF point would output a warning
import warnings
warnings.filterwarnings('ignore')

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

def get_max(arr):
    '''
    Returns the max value and its index in a 1-D npy array
    Parameters
    ----------
    arr: npy array
        1-D array of values
    Returns
    ----------
    ind: int
        index of the max value
    amax: float
        maximum value
    '''
    amax = np.amax(arr)
    ind = np.where(arr == amax)[0]
    
    return ind, amax

if __name__ == "__main__":
    #read input file
    in_fpath = 'sift_ml_input.csv'
    if os.path.isfile(in_fpath):
         ml_input = pd.read_csv(in_fpath, dtype = {'unit_sources': str, 'dart': str,\
                                                 'lat_d': np.float64, 'long_d': np.float64,\
                                                 'extra_forecast': str, 'lat_f': np.float64,\
                                                 'long_f': np.float64})
    else:
        sys.exit("Error: Unit source file cannot be found.")
    
    # Directory to Save plots in
    outdir = "conv_plots_31src_45_300"
    savedir = os.path.join(outdir,'fcast')

    # load unit source dataframes
    dfd = 'unit_src_ts'
    eta_us = {}
    t_us = {}

    extra_forecast = ml_input['extra_forecast'][ml_input.extra_forecast.notnull()].tolist() #Omit CSZ for now

    for name in extra_forecast:
        eta_us[name] = pd.read_csv(os.path.join(dfd,'eta_%s.csv' % name))
        t_us[name] = pd.read_csv(os.path.join(dfd,'t_%s.csv' % name))

    # Load fq data
    npyd = 'npy'
    fcast_eta = np.load(os.path.join(npyd,'fq_fcast_eta.npy'))
    fcast_t = np.load(os.path.join(npyd,'fq_fcast_time.npy'))

    # Load weights
    fq_wts_true = np.load(os.path.join(npyd,'fq_yong_inv_best.npy'))
    fq_wts_inv = [np.load(os.path.join(npyd,'fq_conv1d_wts_test_300.npy')),\
                  np.load(os.path.join(npyd,'fq_conv1d_wts_train_300.npy')),\
                  np.load(os.path.join(npyd,'fq_conv1d_wts_valid_300.npy'))]

    # Load indices, not all used
    inddir = 'indices'
    index = [np.loadtxt(os.path.join(inddir,'fq_dart_test_index.txt')).astype(int),\
                np.loadtxt(os.path.join(inddir,'fq_dart_train_index.txt')).astype(int),\
                np.loadtxt(os.path.join(inddir,'fq_dart_valid_index.txt')).astype(int)]
    runs = [np.loadtxt(os.path.join(inddir,'fq_dart_test_runs.txt')).astype(int),\
           np.loadtxt(os.path.join(inddir,'fq_dart_train_runs.txt')).astype(int),\
           np.loadtxt(os.path.join(inddir,'fq_dart_valid_runs.txt')).astype(int)]
    
    sets = ['test', 'train','valid']
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
        
    if not os.path.isdir(savedir):
        os.mkdir(savedir)
    
    for s, setn in enumerate(sets):
        
        plotdir = os.path.join(savedir,setn)
        if not os.path.isdir(plotdir):
            os.mkdir(plotdir)
        
        ind_tmp = index[s]
        runs_tmp = runs[s]
        wts_tmp = fq_wts_inv[s]
        
        # Extract the relevant runs
        auto_inv = fq_wts_true[ind_tmp,:]
        eta_f = fcast_eta[ind_tmp,:,:]
        t_f = fcast_t[ind_tmp,:,:]

        ntestruns = auto_inv.shape[0]
        nfcastpts = len(extra_forecast)
        npts = 1440

        # eta is in meters, t is in minutes and is in every minute
        auto_inv_eta = np.zeros((ntestruns, nfcastpts, npts))
        auto_inv_t = np.zeros((ntestruns, nfcastpts, npts))
        nn_pred_eta = np.zeros((ntestruns, nfcastpts, npts))
        nn_pred_t = np.zeros((ntestruns, nfcastpts, npts))

        # Calcualte the TS from the weights
        for r in range(ntestruns):
            for n, name in enumerate(extra_forecast):
                auto_inv_eta[r,n,:], auto_inv_t[r,n,:] = calc_ts(auto_inv[r,:], name, eta_us, t_us)
                nn_pred_eta[r,n,:], nn_pred_t[r,n,:] = calc_ts(wts_tmp[r,:], name, eta_us, t_us)

        # Calculate max and time it occurs
        true_max = np.zeros((ntestruns,nfcastpts))
        true_max_i =  np.zeros((ntestruns,nfcastpts))
        auto_max =  np.zeros((ntestruns,nfcastpts))
        auto_max_i =  np.zeros((ntestruns,nfcastpts))
        nn_max = np.zeros((ntestruns,nfcastpts))
        nn_max_i =  np.zeros((ntestruns,nfcastpts))

        # I'm not sure if this is the best way because I also need the index.
        for r in range(ntestruns):
            for n, name in enumerate(extra_forecast):
                true_max_i[r,n], true_max[r,n] = get_max(eta_f[r,n,:])
                auto_max_i[r,n], auto_max[r,n] = get_max(auto_inv_eta[r,n,:])
                nn_max_i[r,n], nn_max[r,n] = get_max(nn_pred_eta[r,n,:])

        # Plot Scatterplots
        fig, axs = plt.subplots(4,2, figsize=(12,24))
        for n, name in enumerate(extra_forecast):
            ax = axs.flatten()[n]

            auto = auto_max[:,n]
            nn = nn_max[:,n]
            true = true_max[:,n]

            evs_auto = np.round(explained_variance_score(true,auto), decimals=2)
            evs_nn = np.round(explained_variance_score(true,nn), decimals=2)

            vmax = max(np.amax(auto), np.amax(nn), np.amax(true))

            ivmax = np.round(1.05*vmax, decimals=2)

            ax.plot([0.0, 1.05*ivmax],
                    [0.0, 1.05*ivmax],
                    "-k",
                    linewidth=0.5)

            line0,= ax.plot( true, 
                            auto,
                            "b.",
                            markersize=2)

            line1, = ax.plot( true, 
                             nn,
                             linewidth=0,
                             marker='D',
                             color='tab:orange',
                             markersize=2)


            legends = [[ line0,   line1],
                       ['Auto Inv., EVS: %s' % evs_auto, 'Neural Net, EVS: %s' % evs_nn]]

            ax.legend(legends[0], legends[1])
            ax.set_xlabel("observed")
            ax.set_ylabel("predicted")
            ax.set_title(name)
            ax.grid(True, linestyle=':')
            ax.set_aspect("equal")

            ax.set_xlim([0.0, ivmax])
            ax.set_ylim([0.0, ivmax])

            if vmax >= 2:
                ax.xaxis.set_ticks(np.arange(0.0, ivmax+1))
                ax.yaxis.set_ticks(np.arange(0.0, ivmax+1))
            elif vmax < 0.4:
                ax.xaxis.set_ticks(np.arange(0.0, ivmax+0.05, 0.05))
                ax.yaxis.set_ticks(np.arange(0.0, ivmax+0.05, 0.05))
            elif vmax < 0.9:
                ax.xaxis.set_ticks(np.arange(0.0, ivmax+0.2, 0.2))
                ax.yaxis.set_ticks(np.arange(0.0, ivmax+0.2, 0.2))
            else:
                ax.xaxis.set_ticks(np.arange(0.0, ivmax+0.25,0.25))
                ax.yaxis.set_ticks(np.arange(0.0, ivmax+0.25,0.25))

        # Temporary since CSZ is missing
        fig.delaxes(axs[3,1])
        fig.show() # not sure if i need this in a script
        plt.tight_layout()
        plt.savefig(os.path.join(savedir, 'scatter_%s.png' % setn))
        fig.clear()
        plt.close(fig)

        # Plot time series
        for r in range(ntestruns):
            fig, axs = plt.subplots(4,2, figsize=(24,16))
            rnum = runs_tmp[r]
            fig.suptitle('Run # %s' % str(rnum), fontsize=24)

            for n, name in enumerate(extra_forecast):
                ax = axs.flatten()[n]

                auto, auto_t = auto_inv_eta[r,n,:], auto_inv_t[r,n,:]
                nn, nn_t = nn_pred_eta[r,n,:], nn_pred_t[r,n,:]
                true, true_t = eta_f[r,n,:], t_f[r,n,:]

                start = np.int(true_max_i[r,n]-60)
                end = np.int(true_max_i[r,n]+60)
                twin = [start, end]
                if start < 0:
                    twin[0] = 0
                if end >= len(true_t):
                    twin[1] = -1

                line0,= ax.plot(true_t[twin[0]:twin[1]], 
                                true[twin[0]:twin[1]],
                                )

                line1, = ax.plot(auto_t[twin[0]:twin[1]], 
                                 auto[twin[0]:twin[1]],
                                )
                line2, = ax.plot(nn_t[twin[0]:twin[1]],
                                 nn[twin[0]:twin[1]]
                                )


                legends = [[ line0,   line1, line2],
                           ['FQ Sol.','Auto Inv.', 'Neural Net']]

                ax.legend(legends[0], legends[1])
                ax.set_xlabel("Time After Earthquake (Minutes)")
                ax.set_ylabel("Amplitude (Meters)")
                ax.set_title(name)

            # Temporary since CSZ is missing
            fig.delaxes(axs[3,1])
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])

            fname = 'fq%s_forecast.png' % str(rnum).zfill(6)
            plt.savefig(os.path.join(plotdir, fname))
            fig.clear()
            plt.close(fig)