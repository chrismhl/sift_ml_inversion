"""
Plot GNSS data for a Cascadia fakequakes realization.
The data comes from
    https://zenodo.org/record/59943
corresponding to the realizations used in this paper:
    http://faculty.washington.edu/rjl/pubs/fakequakes2016/index.html
    
The station locations can be found on the list
    https://earthquake.usgs.gov/monitoring/gps/stations
or the map:
    https://earthquake.usgs.gov/monitoring/gps/Pacific_Northwest
    
The .sac files are read using the ObsPy software,
    https://github.com/obspy/obspy/wiki
    
"""
import numpy as np
import obspy
import os

datadir = 'Cascadia_GNSS\data'
outdir = 'npy'
runnos = np.arange(1300)

stations = ['P316', 'albh', 'bamf', 'bend', 'bils', 'cabl', 'chzz', 'cski', 'ddsn', 
'eliz', 'elsr', 'grmd', 'holb', 'lsig', 'lwck', 'mkah', 'neah', 'nint', 'ntka', 
'ocen', 'onab', 'oylr', 'p154', 'p156', 'p157', 'p160', 'p162', 'p329', 'p343', 
'p362', 'p364', 'p365', 'p366', 'p380', 'p387', 'p395', 'p396', 'p397', 'p398', 
'p401', 'p403', 'p407', 'p435', 'p441', 'p733', 'p734', 'pabh', 'ptrf', 'ptsg', 
'reed', 'sc02', 'sc03', 'seas', 'seat', 'tfno', 'thun', 'till', 'trnd', 'uclu', 
'ufda', 'wdcb', 'ybhb']

#components = ['Z']
components = ['N', 'E', 'Z']
t_len = 512 #seconds

if not os.path.isdir(outdir):
    os.mkdir(outdir)
        
gnss_eta = np.zeros((len(runnos),len(stations)*len(components),t_len))

for r, runno in enumerate(runnos):
    sacdir = os.path.join(datadir, 'cascadia.%s' % str(runno).zfill(6))
    
    channel_index = 0
    for s, station in enumerate(stations):
        for c, comp in enumerate(components):

            gnss_fname = os.path.join(sacdir, '%s.LY%s.sac' % (station,comp))
            LY = obspy.read(gnss_fname)[0]
            gnss_eta[r, channel_index,:] = LY.data 
            
            channel_index = channel_index + 1
    
    print('Processed run %s' % runno)
    
np.save(os.path.join(outdir,'gnss_eta_all.npy'), gnss_eta)
print('GNSS array created')
    
