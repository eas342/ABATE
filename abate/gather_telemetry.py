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


def get_telem(tshirt_path,tserType='spec',smoothingOrder=5):

    if tserType == 'spec':
        tshirt_obj = spec_pipeline.spec(tshirt_path)
        t1, t2 = tshirt_obj.get_wavebin_series(nbins=1)
        x = t1['Time']
        mnemonic = 'IGDP_NRC_A_T_LWFPAH1'
        resultFile = tshirt_obj.specFile
    else:
        tshirt_obj = phot_pipeline.phot(tshirt_path)
        t1, t2 = tshirt_obj.get_tSeries()
        x = t1['Time (JD)']
        mnemonic = 'IGDP_NRC_A_T_SWFPAH1'
        resultFile = tshirt_obj.photFile

    head = fits.getheader(resultFile,extname='ORIG HEADER')
    utc_bjd_offset = (head['MJDMIDI'] - head['BJDMID']) ## days

    mjdStart_utc = Time(np.min(x) + utc_bjd_offset,format='jd') - 0.5 * head['EFFINTTM'] * u.s
    mjdEnd_utc = Time(np.max(x) + utc_bjd_offset,format='jd') + 0.5 * head['EFFINTTM'] * u.s

    api_token = os.environ['MAST_API_TOKEN']
    EDB = engdb.EngineeringDatabase(mast_api_token = api_token)

    result = EDB.timeseries(mnemonic, (mjdStart_utc - 0.5 * u.min).fits, 
                        (mjdEnd_utc + 0.5 * u.min).fits)
    #tel_time_bjd_approx = Time(result.time_mjd,format='mjd') + head['BARTDELT'] * u.s
    tel_time_bjd_approx = Time(result.time_mjd,format='mjd') - utc_bjd_offset * u.day

    poly_tcenter = np.median(tel_time_bjd_approx.mjd)
    fpah_trel = tel_time_bjd_approx.mjd - poly_tcenter
    poly_fpah = phot_pipeline.robust_poly(fpah_trel,np.array(result.value),
                                          smoothingOrder)
    fpah_poly_model = np.polyval(poly_fpah,fpah_trel)
    fpah_poly_interp = np.polyval(poly_fpah,Time(x,format='jd').mjd - poly_tcenter)

    fig, ax = plt.subplots()
    ax.plot(tel_time_bjd_approx.jd,result.value)
    ax.plot(tel_time_bjd_approx.jd,fpah_poly_model)

    ax.axvline(np.nanmin(tel_time_bjd_approx.jd),linestyle='dashed',color='black')
    ax.axvline(np.nanmax(tel_time_bjd_approx.jd),linestyle='dashed',color='black')
    
    make_sure_of_path('plots/telemetry')
    ax.set_xlabel("Time (JD)")
    ax.set_ylabel(mnemonic)
    fig.savefig('plots/telemetry/{}_vs_time.png'.format(mnemonic),bbox_inches='tight',
                dpi=150)
    
    t = Table()
    t['Time'] = x
    t[mnemonic] = fpah_poly_interp
    t[mnemonic + ' diff'] = fpah_poly_interp - np.median(fpah_poly_interp)
    make_sure_of_path('detrending_vectors')
    
    telemfile_path1 = telemfile_path(tshirt_obj.dataFileDescrip)
    t.write(telemfile_path1,overwrite=True)
    