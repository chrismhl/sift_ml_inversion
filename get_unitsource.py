import requests
import os
import numpy as np
import pandas as pd
import sys

def build_url(sources, weights, lat, long, base_url):
    '''
    Extracts the sources and weights from a realization.
    Parameters
    ----------
    sources: list
        Unit source names
    weights: list
        Weights for the corresponding unit sources
    lat: float
        Latitude (N) of buoy/forecast point
    long: float
        Longitude (E) of buoy/forecast point
    base: str
        base url
    Returns
    ----------
    url: str
        URL used for calling NCTR web API
    '''
    weighted_src = []

    for m,src in enumerate(sources):
        weighted_src.append('%s*%s' % (str(weights[m]),src))

    # Building the url for the web API
    unit_source = 'ts=' + '+'.join(weighted_src)

    ltlng = '&lat=%s&lon=%s' % (str(lat),str(long))

    url = base_url + unit_source + ltlng
    
    return url

def get_ts(lat, long, source, authen, verbose=True):
    '''
    Gets and outputs the time series for weighted sum of unit sources at a given latitude and longtitude
    Parameters
    ----------
    lat: float
        Latitude (N) of buoy/forecast point
    long: float
        Longitude (E) of buoy/forecast point
    source: str
        Name of unit source
    authen: Tuple
        Username and password for API authentication
    verbose: bool, default = True
        Boolean for verbose output
    Returns
    ----------
    eta: npy array
        Wave amplitude in meters
    time: npy array
         Time after earthquake corresponding to each entry in eta in minutes.
    '''
    
    # Base URL for web api
    base = 'https://sift.pmel.noaa.gov/websift/prop?'
    
    # Create URL to access Web API
    d_url = build_url([source], [1], lat, long, base)

    if verbose:
        print('Downloading time series from SIFT database URL:')
        print(d_url)

    # Request data form URL.
    r = requests.get(d_url, auth=authen)

    # parse and return as arrays:
    lines = r.text.split('\n')
    time = []
    eta = []
    for line in lines[4:-1]:
        tokens = line.split(' ')
        try:
            time.append(float(tokens[0]))
            eta.append(float(tokens[1]))
        except:
            pass
    time = np.asarray(time) * 60.  # convert to minutes
    eta = np.asarray(eta) / 100.   # convert to meters

    return eta, time

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

if __name__ == "__main__":
    
    # output directory
    outdir = 'unit_src_ts'
    
    # read input file
    in_fpath = 'sift_ml_input.csv'
    if os.path.isfile(in_fpath):
         ml_input = pd.read_csv(in_fpath, dtype = {'unit_sources': str, 'dart': str,\
                                                 'lat_d': np.float64, 'long_d': np.float64,\
                                                 'extra_forecast': str, 'lat_f': np.float64,\
                                                 'long_f': np.float64})
    else:
        sys.exit("Error: Unit source file cannot be found.")
    
    # 31 sources, using Yong's inversions. With exclusions
    unit_sources = ml_input['unit_sources'][ml_input.unit_sources.notnull()].tolist()
    
    if 0:
        # Convert sources to new name
        new_src_names = []
        for m,oldname in enumerate(unit_sources):
            new_src_names.append(oldname[0:2] + oldname[5:] + oldname[4]) 
    
    # DART buoy names
    dart = ml_input['dart'][ml_input.dart.notnull()].tolist()
    
    # Extra forecast points, omitted 'csz' point for now
    extra_forecast = ml_input['extra_forecast'][ml_input.extra_forecast.notnull()].tolist()
    
    # lat long for buoys 
    latlong_d = np.vstack((ml_input['lat_d'][ml_input.lat_d.notnull()].to_numpy(),\
                         ml_input['long_d'][ml_input.long_d.notnull()].to_numpy())).T
    
    # lat long for extra forecast
    latlong_f = np.vstack((ml_input['lat_f'][ml_input.lat_f.notnull()].to_numpy(),\
                         ml_input['long_f'][ml_input.long_f.notnull()].to_numpy())).T
    
    # combine lat long for DARt and forecast
    latlong = np.vstack((latlong_d,latlong_f))
    
    # SIFT user and password:
    authen = tuple(open('authen.txt').readlines()[0].split(',')[:2])
    
    if len(dart+extra_forecast) != len(latlong):
        sys.exit("Error: Number of points and coordinates do not match")
    
    # Create outer directory
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    
    # Get unit source and output as pd DataFrame
    for n,buoy in enumerate(dart+extra_forecast):
        lat = latlong[n,0]
        long = latlong[n,1]
        eta_dict = {}
        t_dict = {}
        
        for src in unit_sources:
            eta, t = get_ts(lat, long, src, authen)
            
            eta_dict[src] = eta
            t_dict[src] = t
       
        eta_df = pd.DataFrame(eta_dict)
        t_df = pd.DataFrame(t_dict)
        fname_e = 'eta_%s.csv' % buoy
        fname_t = 't_%s.csv' % buoy
        
        eta_df.to_csv(os.path.join(outdir,fname_e)\
                      , header=True, index=False)
        print('Saving %s to %s' % (fname_e, outdir))
        t_df.to_csv(os.path.join(outdir,fname_t)\
                   , header=True, index=False)
        print('Saving %s to %s' % (fname_t, outdir))