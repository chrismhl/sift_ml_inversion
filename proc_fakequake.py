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
    name: str
        Source name to be converted
    new: bool, default = True
        Set to True if you are converting from the new format to the old and False
        for the opposite.
    Returns
    ----------
    newname: str
        Converted name
    '''
    
    if new:
        oldname = name[0:2] + 'sz' + name[4] + name[2:4]
        return oldname
    elif not new:
        newname = name[0:2] + name[5:] + name[4]
        return newname

def proc_wts(sources, rnums, outdir):
    '''
    Ingests individual .mat files generated by NCTR containing unit source 
    inversions using the old naming format and outputs a single .npy file. 
    Parameters
    ----------
    sources: list
        Strings of unit source names(old format)
    rnums: npy array
        Run numbers to process
    outdir: str
        filepath of directory containing the .mat files
    Returns
    ----------
    runs_used: list
        Runs where inversions exist
    not_used: list
        Runs where inversions do not exist
    inversions: npy array
        Unit source weights [number of realizations x number of sources]
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
    Converts fakequake time series used as input to the NN from .mat to .npy. 
    Parameters
    ----------
    rnums: list
        Run numbers to convert/process
    gaugenos: list
        Integer DART buoy gauge numbers 
    outdir: str, default='fq_time_series'
        Directory containing the .mat files
    npts: int, default=359
        Length of the time series.
    Returns
    ----------
    eta_all: npy array
        Fakequake amplitude in meters. [realization # x gauge x ts length]
    t_all: npy array
        Time after earthquake corresponding to each entry in eta in minutes. [realization # x gauge x ts length]
    '''
    
    eta_all = np.zeros((len(rnums), len(gaugenos), npts))
    t_all = np.zeros((len(rnums), len(gaugenos), npts))
    
    for r,rnum in enumerate(rnums):        
        for m, gaugeno in enumerate(gaugenos):
            gfile = os.path.join(outdir, 'fq%s_%s.mat' %\
                                 (str(rnum).zfill(6),str(gaugeno).zfill(5)))
            gdata = loadmat(gfile)
            
            # skip t=0 point since it is nan
            t_all[r,m,:]    = gdata['t'].flatten()[1:npts+1]/60  # convert to minutes
            eta_all[r,m,:]  = gdata['h'].flatten()[1:npts+1]/100  # convert to meters
    
    return eta_all, t_all

def proc_fcast_data(rnums, gauges, outdir = 'fq_gauge_waveforms', npts = 1441):
    '''
    Converts extra fakequake time series used for results comparison from .mat to .npy. 
    Parameters
    ----------
     rnums: list
        Run numbers to convert/process
    gaugenos: list
        Integer DART buoy gauge numbers 
    outdir: str
        Directory containing the .mat files
    npts: int, default=1441
        Length of the time series.
    Returns
    ----------
    eta_all: npy array
         Fakequake amplitude in meters. [realization # x gauge x ts length]
    t_all: npy array 
        Time after earthquake corresponding to each entry in eta in minutes. [realization # x gauge x ts length]
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
    Check max eta at all 3 gauges for a given realization and threshold
    Parameters
    ----------
    etas: npy array
        Wave amplitudes [number of gauges x ts length]
    thresh: int
        Threshold to check realization against.
    Returns
    ----------
        True if threshold is met at any 3 gauges, False if it is not
        
    '''
    for i in range(etas.shape[0]):
            if np.amax(etas[i,:]) < thresh:
                return False
            
    return True
    
def out_npy(eta,t, name = 'dart', savedir = 'npy'):
    '''
    Outputs the wave amplitude and time increments as separate .npy files locally.
    Parameters
    ----------
    eta: npy array
        Fakequake amplitude in meters. [realization # x gauge x ts length]
    t: npy array
        Time after earthquake corresponding to each entry in eta in minutes. [realization # x gauge x ts length]
    name: str
        string used in the file name to distinguish between forecast and input time series. 
    savedir: str
        Filepath to the directory the arrays are saved to.
    '''
    if not os.path.isdir(savedir):
        os.mkdir(savedir)

    np.save(os.path.join(savedir,'fq_%s_eta.npy' % name), eta)
    np.save(os.path.join(savedir,'fq_%s_time.npy' % name), t)

# Create indices for training and test sets, need to also add validation    
def shuffle_data(runs, train_size, test_size, seed, inddir = 'indices'):
    '''
    Generates training, validation, and testing sets and outputs run numbers and indices for each set.
    Parameters
    ----------
    runs: list
        Run numbers used.
    train_size: float
        fraction used for training set
    test_size: float
        fraction used for testing set
    seed: int
        random seed used to shuffle data
    inddir: str
        Filepath to the directory the .txt files are saved to
    '''
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
    
    # read input file
    in_fpath = 'sift_ml_input.csv'
    if os.path.isfile(in_fpath):
         ml_input = pd.read_csv(in_fpath, dtype = {'unit_sources': str, 'dart': str,\
                                                 'lat_d': np.float64, 'long_d': np.float64,\
                                                 'extra_forecast': str, 'lat_f': np.float64,\
                                                 'long_f': np.float64})
    else:
        sys.exit("Error: Unit source file cannot be found.")
    
    # Temporarily excluding realization 551.
    mask = np.ones(len(run_range), dtype=bool)
    mask[[551]] = False
    run_exc_551 = run_range[mask,...]
    
    gauges = ml_input['dart'][ml_input.dart.notnull()].tolist()
    
    extra_forecast = ml_input['extra_forecast'][ml_input.extra_forecast.notnull()].tolist() #Omit CSZ for now
    
    # Size of train and test set. Validation set is remainder. Set random seed for reproducibility
    tr_size = 0.7
    ts_size = 0.15
    rseed = 100
    
    # Directory containing the inversions
    wtdir = 'the_best_aut_inv_updated_data'
    
    # List of unit sources used in the old name format and sorted alphabetically.
    ufname = 'unit_sources.csv'
    if os.path.isfile(ufname):
        unit_sources = np.genfromtxt(ufname, dtype = 'str').tolist()  
    else:
        sys.exit("Error: Unit source file cannot be found.") 
    
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
    
    out_npy(eta,time)
    print('FQ DART array created')
    out_npy(eta_f,time_f, name = 'fcast', savedir = 'npy')
    print('FQ forecast array created')
    
    #note the npy directory is created in out_npy()
    np.save(os.path.join('npy','fq_yong_inv_best.npy'),invs) 
    print('Unit source weight array created')

