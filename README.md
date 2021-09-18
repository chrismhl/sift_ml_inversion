# Deep Learning for SIFT Unit Source Inversions
Author: Christopher Liu, University of Washington (UW) Applied Mathematics

NOTE: As of Sept 18, 2021, this README is out of date.

#### Collaborators
* Randy LeVeque, UW Applied Mathematics
* Diego Arcas, NOAA Center for Tsunami Research (NCTR)
* Christopher Moore, NCTR
* Clint Pells, NCTR
* Natalia Sannikova, NCTR
* Vasily Titov, NCTR
* Yong Wei, NCTR

#### Acknowledgements
Work was supported in part by the CICOES Graduate Student Fellowship

## Introduction 
This repository contains a collection of scripts used to,
* Process and download time series data for the fakequakes and unit sources,
* Generate training, validation, and testing data sets,
* Training and predicting with a 1D convolutional neural network,
* Plotting results,

for unit source inversions as a part of NCTR's Short-term Inundation Forecasting for Tsunamis (SIFT) system. Additional documentation can be found in the additional_docs directory.

## Requirements
The Python scripts were made using the following,
* ``python 3.6.13``
* ``conda 4.10.0``
* ``matplotlib``
* ``numpy``
* ``os``
* ``pandas``
* ``pytorch 1.6.0``
* ``requests``
* ``scipy``
* ``sys``

The following were used **only** for plotting results,
* ``sklearn``
* ``clawpack``

## Usage

The scripts **require** the .mat files of the FQ inversions and time series to run and are not included in this repo. The latitude of the centroid of slip for each fakequake realization is required for ``plot_dart_wts.py`` and is also not included in this repo.

### Downloading the unit sources

To obtain the unit source responses at each point of interest run,
```
python get_unitsource.py
```
The lat/long of each point of interest and the list of unit sources is hardcoded into the script. You will also need to include a file named ``authen.txt`` in your current working directory which contains the username and password of the NCTR web API formatted as follows,
```
username,password,
```
The script will then output .csv files with headings corresponding to the unit sources for the wave amplitude and time steps separately at each point of interest.

### Processing .mat files and splitting data
To process the .mat files and generate training, testing, and validation sets run after saving the .mat files to your current working directory and correcting the filepaths in the script as needed,
```
python proc_fakequake.py
```
After running, the script will output the run numbers for each of the 3 sets as well as their corresponding indices. It also outputs npy arrays of the fakequake time series and inverted weights for both the DART buoys and extra forecast points. 

The script will only extract unit sources listed in the ``unit_sources`` list variable. It will also re-order the weights in the order specified in the script which must be consistent with the order in ``get_unitsource.py`` otherwise the plots will be incorrect. The order of the input DART buoys and extra forecast points must also be consistent with the plotting scripts. Note that as of 6/8/21, run 551 and the extra forecast point in the SJdF are excluded. 

### Training, validating, and testing the model
To train, validate, and test the model, open the following Jupyter Notebook,
```
sift_conv1d_train.ipynb
```
Comment out code for plotting and outputting the error if you are satisfied with the model performance. Note that you must re-run the cell where the model is defined if you would like to re-train the model after making adjustments. The last few cells in the notebook give options for outputting the model predictions as well as the model parameters.

### Plotting Results

To plot the comparison of the results at the DART buoys run,
```
python plot_dart_wts.py
```
Example of resulting plots shown below,
![dart](/images/ml_inv_run1267.png)

To plot the comparison of the results at the forecast points run including the scatter plot (one is made for each dataset),
```
python plot_forecast_sift.py
```
Example of resulting plots shown below,
![scatter](/images/scatter_test.png)

![fcast](/images/fq001267_forecast.png)

Note that ``outdir`` in both scripts must be the same. If the script is plotting the wrong things then first check that the order of the unit sources, DART buoys, and forecast gauges is consistent among all scripts
