# Load fakequake time series data, perform thresholding, split into train/validation/test sets
# Load and convert inverted weights.
# Re-written to process .mat files from NCTR instead of GeoClaw output.
# Author: Christopher Liu

import numpy as np
import os
import pandas as pd
from scipy.io import loadmat

def conv_name(name, new = True):
    '''
    Converts names of the subfaults between the new and old scheme.
    Parameters
    ----------
    name:
        String of the name you are converting
    new:
        Set to True if you are converting from the new format to the old and False
        for the opposite.
    Returns
    ----------
    newname:
        String of the converted name
    '''
    
    if new:
        oldname = name[0:2] + 'sz' + name[4] + name[2:4]
        return oldname
    elif not new:
        newname = name[0:2] + name[5:] + name[4]
        return newname

def proc_wts(sources, rnums, outdir):
    '''
    Description....
    Parameters
    ----------
    sources:
        
    rnums:
    
    outdir:
    
    Returns
    ----------
    runs_used:
        
    not_used:
        
    inversions:
        
    '''
    
    # Determine how many runs there are
    tot_runs = len(os.listdir(outdir)) - 1 ### REMOVE -1 LATER
    
    # Create dictionary
    slip = {s : 0 for s in sources}
    
    # Keep track of excluded runs and runs used
    runs_used = []
    not_used = []
    
    inversions = np.zeros((tot_runs,len(sources))) 
    
    n = 0 #index for inversion array
    for rnum in rnums:
        fname = 'fq%s_autoinv_slip.mat' % str(rnum).zfill(6)

        try:
            fpath = os.path.join(outdir,fname)
            mat = loadmat(fpath)
            runs_used.append(rnum)

            faults = mat['faults']
            slips = mat['slip']

            for i in range(len(slips)):
                oldname = conv_name(faults[0][i][0])
                wt = slips[i][0]
                
                if oldname in slip.keys():
                    slip[oldname] = float(wt)
                
            inversions[n,:] = np.array(list(slip.values()))
            
            # Reset dictionary
            slip = {s:0 for s in slip}

            n+=1
        except FileNotFoundError:
            not_used.append(rnum)
        
    return runs_used, not_used, inversions
    
def proc_data(rnums, gaugenos, outdir = 'fq_time_series', npts = 359):
    '''
    Converts .mat to .npy. FQ data for input gauges
    Parameters
    ----------
    rnums:
        
    gaugenos:
        
    outdir:
        
    npts:
        
    Returns
    ----------
    eta_all:
        
    t_all:
        
    '''
    
    eta_all = np.zeros((len(rnums), len(gaugenos), npts))
    t_all = np.zeros((len(rnums), len(gaugenos), npts))
    
    for r,rnum in enumerate(rnums):        
        for m, gaugeno in enumerate(gaugenos):
            gfile = os.path.join(outdir, 'fq%s_%s.mat' %\
                                 (str(rnum).zfill(6),str(gaugeno).zfill(5)))
            gdata = loadmat(gfile)
            
            # skip t=0 point since it is nan
            t_all[r,m,:]    = gdata['t'].flatten()[1:360]/60  # convert to minutes
            eta_all[r,m,:]  = gdata['h'].flatten()[1:360]/100  # convert to meters
    
    return eta_all, t_all

def proc_fcast_data(rnums, gauges, outdir = 'fq_gauge_waveforms', npts = 1441):
    '''
    Converts .mat to .npy. Extra gauges for result comparison.
    Parameters
    ----------
    rnums:
        
    gaugenos:
        
    outdir:
        
    npts:
        
    Returns
    ----------
    eta_all:
        
    t_all:
        
    '''
    
    eta_all = np.zeros((len(rnums), len(gauges), npts))
    t_all = np.zeros((len(rnums), len(gauges), npts))
    
    for r,rnum in enumerate(rnums):        
        for m, name in enumerate(gauges):
            gfile = os.path.join(outdir, '%s_fq%s.mat' %\
                                 (name, str(rnum).zfill(6)))
            gdata = loadmat(gfile)
            
            t_all[r,m,:]    = gdata['t'].flatten()*60  # convert to minutes
            eta_all[r,m,:]  = gdata['h'].flatten()/100  # convert to meters
    
    return eta_all, t_all

def check_thresh(etas, thresh):
    '''
    Check max eta at all 3 gauges for a given realization
    Parameters
    ----------
    etas:
        
    thresh:
        
    Returns
    ----------
        Boolean...
        
    '''
    for i in range(etas.shape[0]):
            if np.amax(etas[i,:]) < thresh:
                return False
            
    return True
    
def out_npy(eta,t, name = 'dart', savedir = 'npy'):
    '''
    Outputs .npy to a file locally. Redo naming 
    Parameters
    ----------
    eta:
        
    t:
    
    name:
    
    savedir:
    
    '''
    if not os.path.isdir(savedir):
        os.mkdir(savedir)

    np.save(os.path.join(savedir,'fq_%s_eta.npy' % name), eta)
    np.save(os.path.join(savedir,'fq_%s_time.npy' % name), t)

# Create indices for training and test sets, need to also add validation    
def shuffle_data(runs, train_size, test_size, seed, inddir = 'indices'):
    np.random.seed(seed)
    set_names = ['train','test']
    dtype = ['index', 'runs']
    nruns = len(runs)
    
    # Calculate size of each set
    tr = int(len(runs) * train_size)
    ts = int(len(runs) * test_size)
    if nruns > (tr+ts):
        vd = nruns - tr - ts
    else:
        vd = 0;
    
    # Create random permutation
    shuffled_ind  = np.arange(nruns)
    np.random.shuffle(shuffled_ind)
    shuffled_rnums = np.array(runs)[shuffled_ind]
    
    # Populate dictonary containing run numbers and indices
    run_ind = {}
    run_ind['train_index'] = shuffled_ind[:tr]
    run_ind['train_runs']= shuffled_rnums[:tr]

    run_ind['test_index'] = shuffled_ind[tr:tr+ts]
    run_ind['test_runs']= shuffled_rnums[tr:tr+ts]
    
    if vd != 0:
        run_ind['valid_index'] = shuffled_ind[tr+ts:]
        run_ind['valid_runs'] = shuffled_rnums[tr+ts:]
        set_names.append('valid')
    
    # Output as .txt files
    if not os.path.isdir(inddir):
        os.mkdir(inddir)
    
    for name in set_names:
        for t in dtype:
            fname = 'fq_dart_%s_%s.txt' % (name, t)
            dict_key = '%s_%s' % (name, t)
            
            np.savetxt(os.path.join(inddir,fname), run_ind[dict_key])   

if __name__ == "__main__":
    run_range = np.arange(0,1300)
    
    # Temporarily excluding realization 551.
    mask = np.ones(len(run_range), dtype=bool)
    mask[[551]] = False
    run_exc_551 = run_range[mask,...]
    
    gauges = [46404,46407,46419]
    extra_forecast = ['anch1', 'anch2', 'dart51407', 'hilo1',\
                      'hilo2', 'sendai1', 'sendai2'] #Omit CSZ for now
    
    # Size of train and test set. Validation set is remainder. Set random seed for reproducibility
    tr_size = 0.7
    ts_size = 0.15
    rseed = 100
    
    # Directory containing the inversions
    wtdir = 'the_best_aut_inv_updated_data'
    
    # List of unit sources used in the old name format and sorted alphabetically.
    unit_sources =['acsza54',\
         'acsza55',\
         'acsza56',\
         'acsza57',\
         'acsza58',\
         'acsza59',\
         'acsza60',\
         'acsza61',\
         'acsza62',\
         'acsza63',\
         'acsza64',\
         'acsza65',\
         'acszb55',\
         'acszb56',\
         'acszb57',\
         'acszb58',\
         'acszb59',\
         'acszb60',\
         'acszb61',\
         'acszb62',\
         'acszb63',\
         'acszb64',\
         'acszb65',\
         'acszz55',\
         'acszz56',\
         'acszz57',\
         'acszz58',\
         'acszz59',\
         'acszz60',\
         'acszz61',\
         'acszz62']
    
    runs_u, runs_e, invs = proc_wts(unit_sources, run_exc_551, wtdir)
    
    eta, time = proc_data(runs_u,gauges)
    eta_f, time_f = proc_fcast_data(runs_u, extra_forecast)
    
    if 0:
        # Filter runs based on max amplitude at input gauges
        filt_runs = []
        for r in runs:
            if check_thresh(eta[r,:,:],0.1):
                filt_runs.append(r)
    
    shuffle_data(runs_u,tr_size, ts_size,rseed)
    
    np.save(r'npy\fq_yong_inv_best.npy',invs)
    out_npy(eta,time)
    out_npy(eta_f,time_f, name = 'fcast', savedir = 'npy')
    

