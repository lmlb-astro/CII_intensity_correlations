import numpy as np
import astropy.io.fits as pyfits
import pandas as pd

#### DEFINE GLOBAL VARIABLES FOR THIS FILE ####

## Spitzer 8 micron bandwidth
bandwidth_8micron = 1.58e13 ## Hz
conv_OrionA = 2.9*4.8e-5*0.2021/10.4*0.74

## [CII] convert K km/s -> erg/s/cm2/sr
conv_CII = 7.0354e-6

########################


#### START OF THE FUNCTION DEFINITIONS ####

## read the data from the fits files, store and store them into a DataFrame
## returns the DataFrame
def read_data_to_DF(path_CII, path_8m, rms_cut, region):
    
    ## get the [CII] data
    hdu = pyfits.open(path_CII)
    data_CII = hdu[0].data
    
    ## get the 8 micron data
    hdu = pyfits.open(path_8m)
    data_8m = hdu[0].data
    
    ## to 1D array
    data_CII = data_CII.ravel()
    data_8m = data_8m.ravel()
    
    ## remove data below [CII] intensity cut
    data_8m = data_8m[data_CII > rms_cut]
    data_CII = data_CII[data_CII > rms_cut]
    
    ## remove all nans from the data
    data_8m = data_8m[~np.isnan(data_CII)]
    data_CII = data_CII[~np.isnan(data_CII)]
    
    data_CII = data_CII[~np.isnan(data_8m)]
    data_8m = data_8m[~np.isnan(data_8m)]
    
    ## convert the 8 micron and [CII] data to erg/s/cm-2/sr
    if(region == "ORIONA"):
        data_8m = np.log10(data_8m * conv_OrionA)
    else:
        data_8m = np.log10(data_8m * bandwidth_8micron / 1e17)
    data_CII = np.log10(data_CII * conv_CII)
    
    ## create a DataFrame
    df = pd.DataFrame({"CII": data_CII, '8 micron': data_8m})
    
    return df


## linear fuction (for fitting)
def lin_func(x, a, b):
    return a*x + b