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
from scipy.spatial.transform import Rotation
from astropy.coordinates import SkyCoord

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

        self.Start_bjd = Time(np.min(x),format='jd')
        self.End_bjd = Time(np.max(x),format='jd')
        self.Start_utc = self.Start_bjd + utc_bjd_offset * u.hr - 0.5 * head['EFFINTTM'] * u.s
        self.End_utc = self.End_bjd + utc_bjd_offset * u.hr + 0.5 * head['EFFINTTM'] * u.s

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
        startTime = (self.Start_utc - extraWindow).fits
        endTime = (self.End_utc + extraWindow).fits

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

def quat_to_celest(quat):
    """
    Given a Quaternion: Return the RA and Dec
    Warning: This was derived from guess and check rather than verifying geometry
    """
    rot = Rotation.from_quat(quat).as_euler('xyz',degrees=True)
    c1 = np.mod(rot[:,2],360)
    c2 = np.mod(-rot[:,1] + 90.,360) - 90.
    c3 = np.mod(rot[:,0],360)
    return c1,c2,c3

def get_slew_hist(tshirt_path,tserType='spec',batchInd=0,
                  showPlot=True,
                  quat_arr_Y=None,
                  quat_arr_X=None,
                  slewRes=None):
    """
    Plot the slew history before and after a time series

    tserType: str
        'spec', 'batchSpec' or 'phot'
    batchInd: int
         use only if it is a batch file to specify which observations/config
    showPlot: bool
        Show a plot of the slew history?
    quat_arr_X: list of numpy arrays
        Saved quaternions X. Skips the telemetry lookup to 
        save time for debugging
    quat_arr_X: list of numpy arrays
        Saved quaternions Y. Skips the telemetry lookup to 
        save time for debugging
    slewRes: Dictionary of telemetry or None
        Dictionary of saved telemetry info. Skips the telemetry lookup to 
        save time for debugging
    """
    esearch = eng_search(tshirt_path=tshirt_path,tserType=tserType,
                        batchInd=batchInd)
    targRA = esearch.head["TARG_RA"]
    targDec = esearch.head["TARG_DEC"]

    #if combQuat is None:
    if (quat_arr_Y is None) | (quat_arr_X is None):
        quat_arr_Y = []
        quat_arr_X = []
        for oneQuat in np.arange(4)+1:
            mnemonic = "SA_ZATTEST{}".format(oneQuat)
            res2 = esearch.get_mnemonic(mnemonic=mnemonic,extraWindow=3. * u.hr)
            #plt.plot(res2.time_mjd, res2.value,label=mnemonic)
            quat_arr_X.append(res2['Time (BJD)'])
            quat_arr_Y.append(res2['Value'])
    combQuat = np.array(quat_arr_Y).T
    coor = quat_to_celest(combQuat)
    coorHist = SkyCoord(coor[0],coor[1],unit=(u.deg,u.deg))
    coordTarg = SkyCoord(targRA,targDec,unit=(u.deg,u.deg))
    sep = coordTarg.separation(coorHist)

    if slewRes is None:
        slewRes =  esearch.get_mnemonic("SA_ZSLEWST",extraWindow=3. * u.hr)


    outT = Table()
    outT['Time (JD)'] = quat_arr_X[0].jd
    outT['Time (MJD)'] = quat_arr_X[0].mjd
    outT['RA'] = coor[0]
    outT['Dec'] = coor[1]
    outT['sep'] = sep.value
    outT.meta['Target RA'] = targRA
    outT.meta['Targ DEC'] = targDec
    outT.meta['expStart'] = esearch.Start_bjd
    outT.meta['expEnd'] = esearch.End_bjd
    
    tNear = sep < 1. * u.deg
    outT.meta['jdArriveNear'] = np.min(outT['Time (JD)'][tNear])
    outT.meta['mjdArriveNear'] = np.min(outT['Time (MJD)'][tNear])
    wait_min = (esearch.Start_bjd.mjd - outT.meta['mjdArriveNear']) * 24. * 60.
    outT.meta['Time Waiting (min)'] = wait_min

    if showPlot == True:
        fig, axArr = plt.subplots(4,sharex=True,figsize=(13,10))
        axArr[0].plot(slewRes['Time (BJD)'].mjd,
                      slewRes['Value'])
        axArr[1].plot(outT['Time (MJD)'],outT['RA'])
        axArr[1].set_ylabel("RA")
        axArr[1].axhline(targRA,color='black',linestyle='dashed')
        axArr[2].plot(outT['Time (MJD)'],outT['Dec'])
        axArr[2].set_ylabel("Dec")
        axArr[2].axhline(targDec,color='black',linestyle='dashed')
        axArr[3].plot(outT['Time (MJD)'],outT['sep'])
        axArr[3].set_ylabel("Distance (deg)")
        axArr[3].set_xlabel("Time (MJD)")
        for oneAx in axArr:
            for oneTime in [esearch.Start_bjd,esearch.End_bjd]:
                oneAx.axvline(oneTime.mjd,color='red',linestyle='dashed')
        axArr[3].axvspan(outT.meta['mjdArriveNear'],
                         esearch.Start_bjd.mjd,alpha=0.5,color='green')


    return outT, slewRes
    #plt.plot(coor[0])
    #plt.axhline(t1.meta['RA'],linestyle='dashed',color='black')
    #plt.plot(coor[1])
    #plt.axhline(t1.meta['DEC'],linestyle='dashed',color='black')