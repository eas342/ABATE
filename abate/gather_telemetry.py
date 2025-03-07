import numpy as np
import matplotlib.pyplot as plt
import os
from astropy.io import fits
#import batman

from copy import deepcopy
import pdb
from astropy.io import fits, ascii
from scipy.optimize import curve_fit
from astropy.table import Table
from astropy.table import vstack
from astropy.time import Time
import astropy.units as u
import astropy.constants as const
from numpy.polynomial import Polynomial
from jwstuser import engdb
from tshirt.pipeline import phot_pipeline, spec_pipeline


def make_sure_of_path(path):
    """
    Check if a path exists
    If not, make it
    """
    if os.path.exists(path) == False:
        os.makedirs(path)


def telemfile_path(descrip):
    telem_fileName = "{}_telem.csv".format(descrip)

    return os.path.join('detrending_vectors',telem_fileName)




class eng_search:
    def __init__(self,tshirt_path,tserType='spec',batchInd=0):
        """
        Gather Mnemonics from MAST engineering database

        """        
        self.tserType = tserType
        if (tserType == 'spec') | (tserType == 'batchSpec'):
            if tserType == 'batchSpec':
                bspec = spec_pipeline.batch_spec(tshirt_path)
                tshirt_obj = bspec.return_spec_obj(batchInd)
            else:
                tshirt_obj = spec_pipeline.spec(tshirt_path)
            t1, t2 = tshirt_obj.get_wavebin_series(nbins=1)
            x = t1['Time']
            
            resultFile = tshirt_obj.specFile
        else:
            tshirt_obj = phot_pipeline.phot(tshirt_path)
            t1, t2 = tshirt_obj.get_tSeries()
            x = t1['Time (JD)']
            
            resultFile = tshirt_obj.photFile
        
        
        self.result = resultFile
        self.x = x
        self.dataFileDescrip = tshirt_obj.dataFileDescrip


        head = fits.getheader(resultFile,extname='ORIG HEADER')
        self.head = head
        utc_bjd_offset = (head['MJDMIDI'] - head['BJDMID']) ## days
        self.utc_bjd_offset = utc_bjd_offset

        self.mjdStart_utc = Time(np.min(x) + utc_bjd_offset,format='jd') - 0.5 * head['EFFINTTM'] * u.s
        self.mjdEnd_utc = Time(np.max(x) + utc_bjd_offset,format='jd') + 0.5 * head['EFFINTTM'] * u.s

        api_token = os.environ['MAST_API_TOKEN']
        self.EDB = engdb.EngineeringDatabase(mast_api_token = api_token)

    def get_mnemonic(self,mnemonic,extraWindow=0.5 * u.min):
        """
        Get the time series for an engineering mnemonic

        Parameters:
        -----------
        mnemonic: str
            Engineering mnemonic
        extraWindow: astropy Time
            How much extra time before exposure start or after exposure end to use?
        """
        startTime = (self.mjdStart_utc - extraWindow).fits
        endTime = (self.mjdEnd_utc + extraWindow).fits

        result = self.EDB.timeseries(mnemonic, startTime,endTime)

        tel_time_bjd_approx = Time(result.time_mjd,format='mjd') - self.utc_bjd_offset * u.day
        outDict = {}
        outDict['Time (BJD)'] = tel_time_bjd_approx
        outDict['Value'] = np.array(result.value)
        return outDict

def get_telem(tshirt_path,tserType='spec',smoothingOrder=5,
              batchInd=0,
              mnemonic=None):
    """
    Gather telemetry for a tshirt object
    
    

    Parameters
    -----------
    tserType: str
        'spec', 'batchSpec' or 'phot'
    batchInd: int
         use only if it is a batch file to specify which observations/config
    """

    esearch = eng_search(tshirt_path=tshirt_path,tserType=tserType,
                         batchInd=batchInd)

    

    if mnemonic is None:
        if (tserType == 'spec') | (tserType == 'batchSpec'):
            mnemonic = 'IGDP_NRC_A_T_LWFPAH1'
        else:
            mnemonic = 'IGDP_NRC_A_T_SWFPAH1'
    res = esearch.get_mnemonic(mnemonic=mnemonic)
    tel_time_bjd_approx = res['Time (BJD)']

    poly_tcenter = np.median(tel_time_bjd_approx.mjd)
    fpah_trel = tel_time_bjd_approx.mjd - poly_tcenter
    poly_fpah = phot_pipeline.robust_poly(fpah_trel,res['Value'],
                                          smoothingOrder)
    fpah_poly_model = np.polyval(poly_fpah,fpah_trel)
    fpah_poly_interp = np.polyval(poly_fpah,Time(esearch.x,format='jd').mjd - poly_tcenter)

    fig, ax = plt.subplots()
    ax.plot(tel_time_bjd_approx.jd,res['Value'])
    ax.plot(tel_time_bjd_approx.jd,fpah_poly_model)

    ax.axvline(np.nanmin(tel_time_bjd_approx.jd),linestyle='dashed',color='black')
    ax.axvline(np.nanmax(tel_time_bjd_approx.jd),linestyle='dashed',color='black')
    
    make_sure_of_path('plots/telemetry')
    ax.set_xlabel("Time (JD)")
    ax.set_ylabel(mnemonic)
    plotFile = 'plots/telemetry/{}_vs_time_{}.png'.format(mnemonic,
                                                         esearch.dataFileDescrip)
    fig.savefig(plotFile,
                bbox_inches='tight',
                dpi=150,facecolor='white')
    
    t = Table()
    t['Time'] = esearch.x
    t[mnemonic] = fpah_poly_interp
    t[mnemonic + ' diff'] = fpah_poly_interp - np.median(fpah_poly_interp)
    make_sure_of_path('detrending_vectors')
    
    telemfile_path1 = telemfile_path(esearch.dataFileDescrip)
    t.write(telemfile_path1,overwrite=True)


# def get_slew_hist(tshirt_path,tserType='spec',batchInd=0,
#                   showPlot=True):
#     """
#     Plot the slew history before and after a time series
#     """
#     esearch = eng_search(tshirt_path=tshirt_path,tserType=tserType,
#                         batchInd=batchInd)