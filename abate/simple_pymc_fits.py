from astropy.io import fits, ascii
from astropy.table import Table
import astropy.units as u
import astropy.constants as const
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from scipy.interpolate import interp1d
import pdb

import exoplanet as xo
import starry
import theano
theano.config.gcc__cxxflags += " -fexceptions"

import pymc3 as pm
import theano.tensor as tt

from tshirt.pipeline import spec_pipeline,phot_pipeline

import pymc3_ext as pmx
import arviz
from celerite2.theano import terms, GaussianProcess
import corner
from scipy import stats
from copy import deepcopy
#from celerite2 import terms, GaussianProcess
import tqdm

## reduce some of the extraneous theano warnings that show up every time
import logging
logging.getLogger('theano').setLevel(logging.ERROR)

from . import gather_telemetry


tshirtDir = spec_pipeline.baseDir

default_paramPath='parameters/spec_params/jwst/sim_mirage_009_grismr_8grp/spec_mirage_009_p013_full_cov_highRN_ncdhasfixed_mpm_refpix.yaml'
default_descrip = 'grismr_002_ncdhasfix'
defaultT0 = 2459492.32, 0.05
default_period = 2.64389803, 0.00000026
default_inc = 86.858, 0.052
default_a = 14.54, 0.14
default_ecc = 'zero'
default_omega = 90.0, 0.01
default_ld = 'quadratic'

default_starry_ld_deg = 6


## Astropy units don't see to carry through, so ...
## Calculate Kepler's 3rd law with a prefactor
prefac = (2. * np.pi / (1.0 * u.day))**2 * (1.0 * u.Rsun)**3 / const.G
unitless_prefac = prefac.to(u.Msun).value


class exo_model(object):
    
    def __init__(self,paramPath=default_paramPath,descrip=default_descrip,t0_lit=defaultT0,
                 recalculateTshirt=True,period_lit=default_period,inc_lit=default_inc,
                 a_lit=default_a,u_lit=None,pipeType='spec',
                 ecc=default_ecc,omega=default_omega,
                 ld_law=default_ld,sigReject=10,
                 starry_ld_degree=default_starry_ld_deg,
                 u_lit_file=None,
                 cores=2,nchains=2,
                 pymc3_init="adapt_full",
                 fitSigma=None,
                 nbins=20,
                 oot_start=0,oot_end=100,
                 trendType='poly',
                 refpixDatFile='',
                 poly_ord=1,
                 legacy_polynomial=False,
                 expStart=None,
                 hstPhase=None,
                 mask=None,
                 offsetMask=None,
                 timeBin=None,
                 wbin_starts=None,
                 wbin_ends=None,
                 nbins_resid=120,
                 ld_start=None,
                 override_times=None,
                 eclipseGeometry="Transit",
                 ror_prior=None,
                 e_depth_guess=1e-3,
                 equalize_bin_err=False,
                 fixLDu1=False,
                 fit_t0_spec=False,
                 fitSinusoid=False,
                 phaseCurveFormulation='standard',
                 correctedBinCenters=False,
                 differentialMode=False,
                 customData=None,
                 batchInd=0,
                 rho_gp_sigma=0.05,
                 expPriorTau = 1e-3,
                 expPriorTauSigma = 2.,
                 ):
        #paramPath = 'parameters/spec_params/jwst/sim_mirage_007_grismc/spec_mirage_007_p015_full_emp_cov_weights_nchdas_mmm.yaml'
        #paramPath = 'parameters/spec_params/jwst/sim_mirage_007_grismc/spec_mirage_007_p016_full_emp_cov_weights_ncdhas_ppm.yaml'
        # paramPath = 'parameters/spec_params/jwst/sim_mirage_009_grismr_8grp/spec_mirage_009_p012_full_cov_highRN_nchdas_ppm_refpix.yaml'
        # descrip = 'grismr_001'
        """
        Set up a starry/exoplanet lightcurve fitting object
        """


        self.paramPath = paramPath
        self.descrip = descrip
        self.t0_lit = t0_lit
        self.inc_lit = inc_lit
        self.a_lit = a_lit
        self.u_lit = u_lit
        self.ecc = ecc
        self.omega = omega
        self.ld_law = ld_law
        self.fixLDu1 = fixLDu1
        self.cores = cores
        self.nchains = nchains
        self.differentialMode = differentialMode
        self.pymc3_init = pymc3_init
        self.e_depth_guess = e_depth_guess
        
        self.eclipseGeometry = eclipseGeometry
        self.ror_prior= ror_prior
        
        if (ld_law == 'quadratic') & (self.eclipseGeometry == 'Transit'):
            ## for quadratic limb darkening transits, 
            ## don't use starry directly, just use exoplanets
            self.starry_ld_degree = None
            self.re_param_u = True
        else:
            self.starry_ld_degree = starry_ld_degree
            self.re_param_u = False
        self.u_lit_file = u_lit_file
        if self.u_lit == 'interpSpecOnly':
            self.set_up_limbdarkening_interpolation()

        if ld_start is None:
            if (ld_law == 'quadratic'):
                self.ld_start = [0.71,0.1]
            else:
                self.ld_start = np.zeros(self.starry_ld_degree) + 0.1
        else:
            self.ld_start = ld_start        
        # paramPath = 'parameters/spec_params/jwst/sim_mirage_009_grismr_8grp/spec_mirage_009_p014_ncdhas_mpm_skip_xsub.yaml'
        # descrip = 'grismr_003_no_xsub'

        #t0_lit = 2459492.8347039, 0.000035
        ## Something was wrong with those, so I'm just putting the center I got from one fit
        
        self.nbins = nbins ## wavelength bins for spectroscopy
        self.paramFile = os.path.join(tshirtDir,paramPath)
        
        self.pipeType = pipeType

        self.batchInd = None ## start with unset
        if pipeType == 'customPhot':
            self.customData = customData
            t1 = Table()
            t1['Time'] = customData['x']
            normVal = np.nanmedian(customData['y'])
            t1['Flux'] = customData['y'] / normVal
            t2 = Table()
            t2['Time'] = customData['x']
            t2['Flux'] = customData['yerr'] / normVal
            timeKey = 'Time'
        elif pipeType == 'phot':
            self.phot = phot_pipeline.phot(self.paramFile)
            t1, t2 = self.phot.get_tSeries()
            timeKey = 'Time (JD)'
            ## normalize
            normVal = np.nanmedian(t1[t1.colnames[1]])
            t1[t1.colnames[1]] = t1[t1.colnames[1]] / normVal
            t2[t2.colnames[1]] = t2[t2.colnames[1]] / normVal
        else:
            if pipeType == 'batchSpec':
                bspec = spec_pipeline.batch_spec(self.paramFile)
                self.batchInd = batchInd
                self.spec = bspec.return_spec_obj(self.batchInd)
            else:
                self.spec = spec_pipeline.spec(self.paramFile)

            t1, t2 = self.spec.get_wavebin_series(nbins=1,recalculate=recalculateTshirt)
            timeKey = 'Time'
        
        print("Raw file search empty is ok for grabbing lightcurves")
        #t1, t2 = spec.get_wavebin_series(nbins=1,specType='Optimal',recalculate=True)
        #t1, t2 = spec.get_wavebin_series(nbins=1,specType='Sum',recalculate=True)

        self.x = np.ascontiguousarray(t1[timeKey],dtype=np.float64)
        self.y = np.ascontiguousarray(t1[t1.colnames[1]] * 1000.) ## convert to ppt
        self.yerr = np.ascontiguousarray(t2[t2.colnames[1]] * 1000.) ## convert to ppt

        self.texp = np.median(np.diff(self.x))
        if mask is None:
            self.mask = np.ones(len(self.x), dtype=bool)
        else:
            self.mask = mask
        self.startMask = deepcopy(self.mask)
        self.offsetMask = offsetMask
        if self.offsetMask is None:
            pass
        else:
            self.steps = np.unique(self.offsetMask)
            self.nSteps = len(self.steps)
        
        self.sigReject = sigReject
        
        # Orbital parameters for the planet, Sanchis-ojeda 2015
        self.period_lit = period_lit

        self.fit_t0_spec = fit_t0_spec
        #t0_lit = 2459491.323591, 0.000035 #GRISMC has different one for some reason
        #t0_lit = 2459560.576056, 0.000035 # GRISMR 
        #t0_lit = 2459560.576056, 0.000035
        #t0_lit = (t0_lit[0] + 0.5705 * period_lit[0], 0.05)
        
        np.random.seed(42) 
    
        # Allow a re-calculation of yerr in case it is under-estimated
        allowedFitSigma = ['oot','fit',None]
        if fitSigma in allowedFitSigma:
            self.fitSigma = fitSigma
        else:
            xtraText = "A common mistake is fitSigma=True instead of fitsSigma='fit'"
            raise Exception('fitSigma={} not in {}. {}'.format(fitSigma,
                                                                     allowedFitSigma,
                                                                     xtraText))
        
        # self.useOOTforError = useOOTforError
        self.oot_start = oot_start
        self.oot_end = oot_end
        
        self.trendType = trendType
        self.refpixDatFile = refpixDatFile
        self.poly_ord = poly_ord
        self.legacy_polynomial = legacy_polynomial
        
        self.rho_gp_sigma = rho_gp_sigma

        self.expStart = expStart
        self.expPriorTau = expPriorTau
        self.expPriorTauSigma = expPriorTauSigma
        self.hstPhase = hstPhase

        self.timeBin = timeBin
        self.override_times = override_times
        
        self.correctedBinCenters = correctedBinCenters
        self.wbin_starts = wbin_starts
        self.wbin_ends = wbin_ends
        self.check_wavebins()

        self.nbins_resid = nbins_resid
        
        self.equalize_bin_err = equalize_bin_err
        if ((self.timeBin is not None) & (self.equalize_bin_err == False)):
            print("NOTE NOTE NOTE ************** NOTE NOTE NOTE NOTE")
            print("Binning with and equalize_bin_err=False not recommended")
            print("NOTE NOTE NOTE ************** NOTE NOTE NOTE NOTE")

        self.broadband_fit_file = 'fit_results/broadband_fit_{}.csv'.format(self.descrip)
        if (pipeType == 'spec') | (pipeType == 'batchSpec'):
            self.specFileName = os.path.join('fit_results',self.descrip,'spectrum_{}.csv'.format(self.descrip))
        else:
            self.specFileName = None
        
        self.fitSinusoid = fitSinusoid
        self.phaseCurveFormulation = phaseCurveFormulation

        if 'fpah' in self.trendType:
            self.get_telem_vec()
        if 'refpix' in self.trendType:
            self.get_refpix_vec()
    
    def get_telem_vec(self,reDo=False):
        """
        Get a vector from spacecraft telemetry
        'e.g.' Focal plane housing array temperature

        Parameters
        -----------
        reDo: bool
            Re-do the telemetry vector?
        """
        if (self.pipeType == 'spec') | (self.pipeType == 'batchSpec'):
            fileDescrip = self.spec.dataFileDescrip
        elif self.pipeType == 'phot':
            fileDescrip = self.phot.dataFileDescrip
        else:
            raise Exception("Telemetry not set up for pipType={}".format(self.pipeType))
        
        pathName = gather_telemetry.telemfile_path(fileDescrip)
        if (os.path.exists(pathName) == False) | (reDo==True):
            gather_telemetry.get_telem(self.paramPath,tserType=self.pipeType,
                                       batchInd=self.batchInd)
        dat = ascii.read(pathName)
        
        if self.pipeType == 'phot':
            self.fpah = np.array(dat['IGDP_NRC_A_T_SWFPAH1 diff'])
        else:
            self.fpah = np.array(dat['IGDP_NRC_A_T_LWFPAH1 diff'])
    
    def get_refpix_vec(self):
        """
        Gather the reference pixel telemetry
        """
        if os.path.exists(self.refpixDatFile) == False:
            raise Exception("Could not find refpix file {}".format(self.refpixDatFile))
        else:
            refpixDat = ascii.read(self.refpixDatFile)
        
        self.refpix = refpixDat['mean refpix'] - np.median(refpixDat['mean refpix'])

    def check_phase(self):
        phase = (self.x - self.t0_lit[0]) / self.period_lit[0]
        return phase

    def check_wavebins(self):
        """
        Check for some mistakes in the wavebins
        """
        if self.wbin_starts is None:
            if self.wbin_ends is not None:
                raise Exception("Can't only set wbin Starts to None")
        elif self.wbin_ends is None:
            if self.wbin_starts is not None:
                raise Exception("Can't only set wbin Ends to None")
        else:
            binWidths = (np.array(self.wbin_ends) - 
                         np.array(self.wbin_starts))
            if np.sum(binWidths == 0) > 0:
                raise Exception("Some Bin widths are zero")


    def inspect_physOrb_params(self,paramVal):
        """
        Figure out if the supplied physical or orbital parameter
        is fixed or a variable.

        Parameters
        ----------
        paramVal: string name of parameter
            The name of the physical or orbital parameter
        
        Returns
        -------
        paramLen, int
            The length of the value. 1 is fixed, 2 is variable
        """
        if (type(paramVal) == float) | (type(paramVal) == int) | (type(paramVal) == np.float64):
            paramLen = 1
        else:
            paramLen = len(paramVal)
        return paramLen

    def set_up_limbdarkening_interpolation(self):
        """
        Set up the interpolation functions from the 
        saved file
        """
        dat = ascii.read(self.u_lit_file)
        self.u_lit_degree = dat.meta['degree']
        interpFuncs = []
        for u_lit_ind in range(self.u_lit_degree):
            thisInterp = interp1d(dat['wave'],
                                  dat['u{}'.format(u_lit_ind+1)])
            interpFuncs.append(thisInterp)
        self.interpFuncs = interpFuncs
        assert self.ld_law == dat.meta['law'],'u_lit and used ld law are different'


    def build_model(self, specInfo=None):
        """
        Build a pymc3 model
        
        specInfo: dict
            Spectroscopic info if it's a spectroscopic wavelength bin
        """
        
        mask = self.mask
        with pm.Model() as model:
            
            # Parameters for the stellar properties
            mean = pm.Normal("mean", mu=1000., sd=10,testval=1000.)


            if (self.u_lit == 'interpSpecOnly'):
                if specInfo == None:
                    ## Broadband fit the limb dark parameters
                    fitLimbDarkParamsOverride = True
                else:
                    fitLimbDarkParamsOverride = False
            else:
                fitLimbDarkParamsOverride = False

            if (self.fixLDu1 == True) & (specInfo is not None):
                u_star = 'special'
            elif (self.u_lit == None) | (fitLimbDarkParamsOverride == True):
                if (self.eclipseGeometry == 'Transit') | (self.eclipseGeometry == 'PhaseCurve'):
                    if self.ld_law == 'quadratic':
                        u_star = xo.QuadLimbDark("u_star",testval=self.ld_start)
                    else:
                        # u_star = pm.Lognormal("u_star",mu=np.log(0.1), sigma=0.5,
    #                                             shape=(default_starry_ld_deg,))
                        #ld_start[0] = 0.1
                        ld_start = self.ld_start
                        u_star = pm.Normal("u_star",mu=ld_start,testval=ld_start,
                                           sigma=2.0,
                                           shape=(self.starry_ld_degree,))
                else:
                    u_star = 0.0
            elif (self.u_lit == 'interpSpecOnly'):
                u_star = []
                for u_lit_ind in range(self.u_lit_degree):
                    thisWave = specInfo['waveMid']
                    u_star.append(self.interpFuncs[u_lit_ind](thisWave))
            else:
                u_star = self.u_lit
        
            if specInfo == None:
                ## ie broadband lightcurve

                if self.inspect_physOrb_params(self.a_lit) == 1:
                    a = self.a_lit
                else:
                    a = pm.Normal("a",mu=self.a_lit[0],sigma=self.a_lit[1],testval=self.a_lit[0])
                
                if self.inspect_physOrb_params(self.inc_lit) == 1:
                    incl = self.inc_lit
                else:
                    incl = pm.Normal("incl",mu=self.inc_lit[0],sigma=self.inc_lit[1],testval=self.inc_lit[0])
                # BoundedNormal = pm.Bound(pm.Normal, lower=0, upper=3)
                # m_star = BoundedNormal(
                #     "m_star", mu=M_star[0], sd=M_star[1]
                # )
                # r_star = BoundedNormal(
                #     "r_star", mu=R_star[0], sd=R_star[1]
                # )
                
                if self.inspect_physOrb_params(self.period_lit) == 1:
                    period = self.period_lit
                else:
                    period = pm.Normal("period",mu=self.period_lit[0],sd=self.period_lit[1])
                
                if self.inspect_physOrb_params(self.t0_lit) == 1:
                    t0 = self.t0_lit
                else:
                    t0 = pm.Normal("t0", mu=self.t0_lit[0], sd=self.t0_lit[1])
                #t0 = t0_lit[0]
                
                if (self.eclipseGeometry == 'Eclipse') | (self.eclipseGeometry == 'PhaseCurve'):
                    e_depth = pm.Normal("e_depth",mu=self.e_depth_guess,sigma=2e-3)
                    if self.ror_prior is None:
                        raise Exception("Must have an ror prior for eclipse")
                    ror = pm.TruncatedNormal("ror",mu=self.ror_prior[0],
                            testval=self.ror_prior[0],
                            sigma=self.ror_prior[1],
                            lower=0.0)
                else:
                    ror = pm.Lognormal("ror", mu=np.log(0.0822), sigma=0.5)
                
                #ror = pm.Deterministic("ror", tt.pow(10,logr_pl) / R_star[0])#r_star)
                #b = xo.distributions.ImpactParameter("b", ror=ror)
                ecc_from_broadband = False
                
                if hasattr(self,'x_full_res'):
                    x_in, y_in, yerr_in = self.x_full_res, self.y_full_res, self.yerr_full_res
                else:
                    x_in, y_in, yerr_in = self.x, self.y, self.yerr
                
                specModel = False
                waveName = 'broadband'


            else:
                broadband = specInfo['broadband']
                if self.inspect_physOrb_params(self.a_lit) == 1:
                    a = self.a_lit
                else:
                    a = get_from_t(broadband,'a','mean')
                # a = pm.Normal("a",mu=get_from_t(broadband,'a','mean'),
#                               sigma=get_from_t(broadband,'a','std'))
                if self.inspect_physOrb_params(self.inc_lit) == 1:
                    incl = self.inc_lit
                else:
                    incl = get_from_t(broadband,'incl','mean')
                # incl = pm.Normal("incl",mu=get_from_t(broadband,'incl','mean'),
#                                  sigma=get_from_t(broadband,'incl','std'))
                if self.inspect_physOrb_params(self.period_lit) == 1:
                    period = self.period_lit
                else:
                    period = get_from_t(broadband,'period','mean')
                #period = pm.Normal("period",mu=self.period_lit[0],sd=self.period_lit[1])
                
                if self.inspect_physOrb_params(self.t0_lit) == 1:
                    t0_broadband = self.t0_lit
                else:
                    t0_broadband = get_from_t(broadband,'t0','mean')
                
                if self.fit_t0_spec == True:
                    t0 = pm.Normal("t0",mu=t0_broadband,sd=self.t0_lit[1])
                else:
                    t0 = t0_broadband
                # t0 = pm.Normal("t0", mu=get_from_t(broadband,'t0','mean'),
                #                  sigma=get_from_t(broadband,'t0','std'))
                #t0 = t0_lit[0]
                
                if (self.eclipseGeometry == 'Eclipse') | (self.eclipseGeometry == 'PhaseCurve'):
                    e_depth = pm.Normal("e_depth",mu=1e-3,sigma=2e-3)
                    if self.ror_prior is None:
                        raise Exception("Must have an ror prior for eclipse")
                    ror = pm.TruncatedNormal("ror",mu=self.ror_prior[0],
                                             sigma=self.ror_prior[1],
                                             testval=self.ror_prior[0],
                                             lower=0.0)
                else:
                     mean_r = get_from_t(broadband,'ror','mean')
                     ror = pm.Lognormal("ror", mu=np.log(mean_r), sigma=0.3)
                
                if 'ecc' in broadband['var name']:
                    ecc_from_broadband = True
                    ecc_from_broadband_val = get_from_t(broadband,'ecc','mean')
                    # ecc_from_broadband_val = (get_from_t(broadband,'ecc','mean'),
                    #                           get_from_t(broadband,'ecc','std'))
                    
                else:
                    ecc_from_broadband = False
                    ecc_from_broadband_val = (np.nan,np.nan)
                
                if self.fixLDu1 == True:
                    assert self.ld_law=='quadratic'
                    ## Make sure u1 is fixed
                    u1_use = get_from_t(broadband,'u_star__0','mean')
                    ## Kipping et al. 2013 equation 8
                    u2_use = pm.Uniform('u_star__1',lower=-0.5 * u1_use,upper=1. - u1_use)
                    
                    #u1 = pm.Deterministic('u_star__0',u1_use)
                    
                    u_star = [u1_use,u2_use]

                
                x_in, y_in, yerr_in = specInfo['x'], specInfo['y'], specInfo['yerr']
                specModel = True
                waveName = specInfo['waveName']
            
            if self.override_times is None:
                pass
            else:
                x_in = self.override_times
            
            if 'fpah' in self.trendType:
                if hasattr(self,'fpah_full_res'):
                    fpah_in = self.fpah_full_res
                else:
                    fpah_in = self.fpah
            if 'refpix' in self.trendType:
                if hasattr(self,'refpix_full_res'):
                    refpix_in = self.refpix_full_res
                else:
                    refpix_in = self.refpix
            
            if self.timeBin is None:
                ## no binning of lightcurves
                x, y, yerr = x_in, y_in, yerr_in
                if 'fpah' in self.trendType:
                    fpah = fpah_in
                if 'refpix' in self.trendType:
                    refpix = refpix_in
            else:
                if (self.timeBin < len(mask)) | hasattr(self,'full_res_mask'):
                    ### Handles two cases
                    ## 1) the mask is for full time resolution
                    ## 2) there is a saved separate mask for full time resolution
                    if hasattr(self,'full_res_mask'):
                        pass
                    else:
                        self.full_res_mask = deepcopy(mask)


                    x_to_bin = x_in[self.full_res_mask]
                    y_to_bin = y_in[self.full_res_mask]

                    if 'fpah' in self.trendType:
                        fpah_to_bin = fpah_in[self.full_res_mask]
                    if 'refpix' in self.trendType:
                        refpix_to_bin = refpix_in[self.full_res_mask]
                    
                    ## do nothing if the mask is meant for binned data
                    if (self.timeBin == len(mask)):
                        pass
                    else:
                        ## if the mask is not meant for binned data,
                        ## include all points
                        mask = np.ones(self.timeBin,dtype=bool)
                        self.mask = mask
                        self.startMask = deepcopy(mask)
                    

                else:
                    x_to_bin = x_in
                    y_to_bin = y_in
                    if 'fpah' in self.trendType:
                        fpah_to_bin = fpah_in
                    if 'refpix' in self.trendType:
                        refpix_to_bin = refpix_in    

                x, y, yerr = phot_pipeline.do_binning(x_to_bin, y_to_bin,nBin=self.timeBin)
                y = np.ascontiguousarray(y)
                yerr = np.ascontiguousarray(yerr)
                if self.equalize_bin_err == True:
                    yerr = np.ones_like(yerr) * np.nanmedian(yerr)
                
                if 'fpah' in self.trendType:
                    xfpah,fpah,fpah_err = phot_pipeline.do_binning(x_to_bin,
                                                                fpah_to_bin,
                                                                nBin=self.timeBin)
                    fpah = np.ascontiguousarray(fpah)
                if 'refpix' in self.trendType:
                    xrefpix,refpix,refpix_err = phot_pipeline.do_binning(x_to_bin,
                                                                refpix_to_bin,
                                                                nBin=self.timeBin)
                    refpix = np.ascontiguousarray(refpix)
                
                finite_y = np.isfinite(y)
                
                mask = mask & finite_y ## make sure to only include finite points
                ## If there are gaps, they will make NaNs that give NaN for the likelihood
                ## unless they are masked out
                
                if specInfo == None:
                    ## for brandband lightcurves
                    if hasattr(self,'x_full_res'):
                        pass ## full resolution is already saved
                    else:
                        self.x_full_res = deepcopy(self.x)
                        self.y_full_res = deepcopy(self.y)
                        self.yerr_full_res = deepcopy(self.yerr)
                        if 'fpah' in self.trendType:
                            self.fpah_full_res = deepcopy(self.fpah)
                        if 'refpix' in self.trendType:
                            self.refpix_full_res = deepcopy(self.refpix)

                    self.x = x
                    self.y = y
                    self.yerr = yerr
                    if 'fpah' in self.trendType:
                        self.fpah = fpah
                    if 'refpix' in self.trendType:
                        self.refpix = refpix
                
                if self.offsetMask is not None:
                    if hasattr(self,'full_res_offsetMask') == True:
                        pass
                    else:
                        self.full_res_offsetMask = deepcopy(self.offsetMask)
                        offsetMask_x, offsetMask_y, offsetMask_yerr = phot_pipeline.do_binning(self.x_full_res,
                                                                                               self.full_res_offsetMask,
                                                                                               nBin=self.timeBin)
                        offsetMask_binned = np.array(offsetMask_y,dtype=int)
                        self.offsetMask = offsetMask_binned
                        ## mask out points with a mixture between the two steps
                        unMixedPoints = np.mod(offsetMask_y,1.0) == 0
                        mask = mask & unMixedPoints
                
            # if self.useOOTforError == True:
#                 yerr = np.std(y[self.oot_start:self.oot_end]) * np.ones_like(yerr)
            if self.fitSigma == 'fit':
                sigma_lc = pm.Lognormal("sigma_lc", mu=np.log(np.median(yerr)), sigma=0.5)
            elif self.fitSigma == 'oot':
                sigma_lc = np.std(y[self.oot_start:self.oot_end])
            else:
                sigma_lc = yerr[mask]
            
            if self.eclipseGeometry == 'Eclipse':
                depth = pm.Deterministic('depth',e_depth * 1.0)
            else:
                depth = pm.Deterministic('depth',tt.pow(ror,2))
            
            if self.ecc == 'zero':
                ecc = 0.0
                omega = 90.
            elif ecc_from_broadband == True:
                omega = get_from_t(broadband,'omega','mean')
                ecc = ecc_from_broadband_val
            elif self.ecc == 'free':
                ecs = pmx.UnitDisk("ecs", testval=np.array([0.01, 0.0]))
                ecc = pm.Deterministic("ecc", tt.sum(ecs ** 2))
                omega = pm.Deterministic("omega", tt.arctan2(ecs[1], ecs[0]))
                # omega = pm.Normal("ecc",mu=ecc_from_broadband_val[0],sigma=ecc_from_broadband_val[1])
            else:
                #BoundedNormal = pm.Bound(pm.Normal, lower=0.0, upper=1.0)
                if self.inspect_physOrb_params(self.ecc) == 1:
                    ecc = self.ecc
                else:
                    ecc = pm.TruncatedNormal("ecc",mu=self.ecc[0],sigma=self.ecc[1],
                                            lower=0.0,upper=1.0,testval=self.ecc[0])
                if self.inspect_physOrb_params(self.omega) == 1:
                    omega = self.omega
                else:
                    omega = pm.Normal("omega",mu=self.omega[0],sigma=self.omega[1],
                                      testval=self.omega[0])
            # xo.eccentricity.kipping13("ecc_prior", fixed=True, observed=ecc)
        
            # ecc = 0.0
            # omega = np.pi/2.0
            # xo.eccentricity.kipping13("ecc_prior", fixed=True, observed=ecc)
            
            if (self.ld_law == 'quadratic') & (self.eclipseGeometry == "Transit"):
                # Orbit model
                orbit = xo.orbits.KeplerianOrbit(
                    period=period,
                    a=a,
                    incl=incl * np.pi/180.,
                    t0=t0,
                    ecc=ecc,
                    omega=omega * np.pi/180.,
                )
        
                light_curves_obj = xo.LimbDarkLightCurve(u_star)
                light_curves1 = light_curves_obj.get_light_curve(orbit=orbit, r=ror,
                                                                 t=x, texp=self.texp)
        
                light_curve = (tt.sum(light_curves1, axis=-1) + 1.)
            
                
            else:
                
                
                Msys = unitless_prefac * a**3 / period**2
                m_star = Msys ## assume it's all star for now
                
                star_map = starry.Map(udeg=self.starry_ld_degree)
                star_map[1:] = u_star
                star = starry.Primary(star_map,m=m_star,r=1.0)
                
                if self.eclipseGeometry == 'Transit':
                    amp = 0
                else:
                    amp = e_depth
                
                planet = starry.kepler.Secondary(starry.Map(amp=amp),
                                                 m=0.0,
                                                 r=ror,
                                                 porb=period,
                                                 prot=period,
                                                 t0=t0,
                                                 ecc=ecc,
                                                 omega=omega,
                                                 inc=incl)
                sys = starry.System(star,planet,texp=self.texp)
                light_curve = sys.flux(t=x)
                self.sys = sys
                
                # ## make sure that the limb darkening law is physical
                # is_physical = pm.math.eq(star_map.limbdark_is_physical(), 1)
                # switch = pm.math.switch(is_physical,-np.inf,0)
                # ## Assign a potential to avoid these maps
                # physical_LD = pm.Potential('physical_ld', switch)
                
                # # Add a constraint that the map should be non-negative
                # mu = np.linspace(0,1,30)
                # map_evaluate = star_map.intensity(mu=mu)
                # ## number of points that are less than zero
                # num_bad = pm.math.sum(pm.math.lt(map_evaluate,0))
                # ## check if there are any "bad" points less than zero
                # badmap_check = pm.math.gt(num_bad, 0)
                # ## Set log probability to negative infinity if there are bad points. Otherwise set to 0.
                # switch = pm.math.switch(badmap_check,-np.inf,0)
                # ## Assign a potential to avoid these maps
                # nonneg_map = pm.Potential('nonneg_map', switch)
            

            if (self.fitSinusoid == True) & (self.phaseCurveFormulation != 'old-legacy'):
                phaseAmp = pm.TruncatedNormal('phase_amp',mu=0.5,sigma=0.5,
                                              lower=0.0,testval=1e-5)
                phaseOffset = pm.Normal('phaseOffset',mu=0,sigma=50,testval=0.0)
                arg = (x - t0) * np.pi * 2. / period + np.pi * phaseOffset/180.
                phaseModel = 1.0 - phaseAmp - phaseAmp * tt.cos(arg)
                ## save a variable for the phase variations to see them separately
                phaseModelVar = pm.Deterministic('phaseModel',phaseModel)

                if self.phaseCurveFormulation == 'standard':
                    ## Find a dark planet model as a reference
                    planet_dark = starry.kepler.Secondary(starry.Map(amp=0),
                                        m=0.0,
                                        r=ror,
                                        porb=period,
                                        prot=period,
                                        t0=t0,
                                        ecc=ecc,
                                        omega=omega,
                                        inc=incl)
                    sys2 = starry.System(star,planet_dark,texp=self.texp)
                    light_curve_darkplanet = sys2.flux(t=x)
                    light_curve_darkplanet_var = pm.Deterministic('lc-darkplanet',light_curve_darkplanet)
                    planet_emission = light_curve - light_curve_darkplanet
                    light_curve = light_curve_darkplanet + planet_emission * phaseModel
                    nightside_f = pm.Deterministic('nightsideDepth',e_depth * (1. - 2. * phaseAmp))
                    planet_amp = pm.Deterministic('planetFullamp',e_depth * 2. * phaseAmp)
                elif self.phaseCurveFormulation == 'approx':
                    light_curve = light_curve - amp + amp * phaseModel
                else:
                    raise Exception("{} not recognized".format(self.phaseCurveFormulation))
            ## save a "light_curves" pymc3 varaible with the astrophysical lightcurve
            light_curves = pm.Deterministic("light_curves",light_curve)

            lc_trend_vector_multiplier = mean
            if 'poly' in self.trendType:
                xNorm = (x - np.median(x))/(np.max(x) - np.min(x))
                #xNorm_var = pm.Deterministic("xNorm",xNorm)
                poly_coeff = pm.Normal("poly_coeff",mu=0.0,testval=0.0,
                                       sigma=0.1,
                                       shape=(self.poly_ord))
                # if self.poly_ord > 1:
                #     raise NotImplementedError("Only does linear for now")
                for poly_ind in np.arange(self.poly_ord):
                    if poly_ind == 0:
                        poly_eval = xNorm * poly_coeff[self.poly_ord - poly_ind - 1]
                    else:
                        poly_eval = (poly_eval + poly_coeff[self.poly_ord - poly_ind - 1]) * xNorm
                full_coeff = poly_coeff# np.append(poly_coeff,0)

                if self.legacy_polynomial == True:
                    lc_trend_vector_adder = lc_trend_vector_adder + poly_eval
                else:
                    lc_trend_vector_multiplier = lc_trend_vector_multiplier * (1.0 + poly_eval)
            
            if 'fpah' in self.trendType:
                telemCoeff = pm.Normal('fpahCoeff',mu=0,sigma=5e-3)
                lc_trend_vector_multiplier = lc_trend_vector_multiplier * (1.0 + fpah * telemCoeff)
            if 'refpix' in self.trendType:
                refpixCoeff = pm.Normal('refpixCoeff',mu=0,sigma=0.5)
                lc_trend_vector_multiplier = lc_trend_vector_multiplier * (1.0 + refpix * refpixCoeff)
            
            if self.offsetMask is None:
                pass
            else:
                step_offsets = 0.0
                steps = self.steps
                nSteps = self.nSteps
                if nSteps == 1:
                    pass
                else:
                    offsetArr = pm.Normal('stepOffsets',mu=0.0,testval=0.0,sigma=1,
                                        shape=(nSteps-1))
                    for oneStep in steps:
                        if oneStep == 0:
                            pass
                        else:
                            pts = self.offsetMask == oneStep
                            step_offsets = step_offsets + tt.switch(pts,offsetArr[oneStep-1],0.0)
                            lc_trend_vector_multiplier = lc_trend_vector_multiplier * (1.0 + step_offsets * 1e-3)
                            #light_curves_trended[pts] = light_curves_trended[pts] + offsetArr[oneStep-1]
            
            light_curves_trended = pm.Deterministic("lc_trended",(light_curve * lc_trend_vector_multiplier))

            if (self.fitSinusoid == True) & (self.phaseCurveFormulation == 'old-legacy'):
                phaseAmp = pm.TruncatedNormal('phase_amp',mu=1e-5,sigma=1.0,
                                              lower=0.0,testval=1e-5)
                phaseOffset = pm.Normal('phaseOffset',mu=0,sigma=50,testval=0.0)
                arg = (x - t0) * np.pi * 2. / period + np.pi * phaseOffset/180.
                phaseModel = 1.0 - phaseAmp * tt.cos(arg)
                ## save a variable for the phase variations
                phaseModelVar = pm.Deterministic('phaseModel',phaseModel)
                light_curves_semifinal = light_curves_trended * phaseModel

            if self.expStart == True:
                expTau = pm.Lognormal("exp_tau",mu=np.log(self.expPriorTau),
                                      sd=self.expPriorTauSigma)
                expAmp = pm.Normal("exp_amp",mu=1e-3,sd=1e-2)
                exp_eval = (1. - expAmp * tt.exp(-(x - np.min(x)) / expTau))
                exp_model = pm.Deterministic('exp_model',exp_eval)
                light_curves_semifinal = light_curves_trended * exp_model
            elif self.expStart == 'HST':
                expTau = pm.Lognormal("exp_tau",mu=np.log(self.expPriorTau),
                                      sd=self.expPriorTauSigma)
                expAmp = pm.Normal("exp_amp",mu=1e-3,sd=1e-2)
                x_exp = self.hstPhase

                exp_eval = (1. - expAmp * tt.exp(-(x_exp) / expTau))
                exp_model = pm.Deterministic('exp_model',exp_eval)
                light_curves_semifinal = light_curves_trended * exp_model
                
            elif self.expStart == 'double':
                expTau1 = pm.Lognormal("exp_tau1",mu=np.log(1e-3),sd=2)
                expAmp1 = pm.Normal("exp_amp1",mu=1e-3,sd=1e-2)
                expTau2 = pm.Lognormal("exp_tau2",mu=np.log(1e-3),sd=2)
                expAmp2 = pm.Normal("exp_amp2",mu=1e-2,sd=1e-2)
                xrel = x - np.min(x)
                exp_eval = (1. - expAmp1 * tt.exp(-(xrel) / expTau1) - expAmp2 * tt.exp(-(xrel) / expTau2))
                exp_model = pm.Deterministic('exp_model',exp_eval)
                light_curves_semifinal = light_curves_trended * exp_model
            else:
                light_curves_semifinal = light_curves_trended
            
            if (self.differentialMode == True) & (specInfo is not None):
                light_curves_semifinal = light_curves_semifinal/specInfo['BBmxap_soln']['lc_final']
            ## the mean converts to parts per thousand
        
            # resid = self.y[mask] - light_curve
            #
            # # Transit GP parameters
            #
            # # original ones
            # # sigma_lc = pm.Lognormal("sigma_lc", mu=np.log(np.std(self.y[mask])), sd=10)
            # # rho_gp = pm.Lognormal("rho_gp", mu=0, sd=10)
            # # sigma_gp = pm.Lognormal("sigma_gp", mu=np.log(np.std(self.y[mask])), sd=10)
            #
            # ## adjusted set 1
            # # sigma_lc = pm.Lognormal("sigma_lc", mu=-3 * np.log(10.), sigma=2,testval=1e-3 * np.log(10.))
            # # rho_gp = pm.Lognormal("rho_gp", mu=0, sigma=10)
            # # sigma_gp = pm.Lognormal("sigma_gp", mu=-3 * np.log(10.), sigma=2,testval=1e-3 * np.log(10.))
            #
            # # GP model for the light curve
            # ## adjusted set 2
            # sigma_lc = pm.Lognormal("sigma_lc", mu=np.log(np.std(self.y[mask])), sigma=0.5)
            # ## the correlations are on 0.02 day timescales
            # rho_gp = pm.Lognormal("rho_gp", mu=np.log(1e-2), sigma=0.5)
            #
            if 'gp' in self.trendType:
            ## assume a similar order of magnitude as the errorbars
                sigma_gp = pm.Lognormal("sigma_gp", mu=np.log(np.median(yerr)), sigma=0.5,
                                        testval=np.median(yerr) * 0.5)
                rho_gp_guess = (np.max(self.x) - np.min(self.x))
                rho_gp = pm.Lognormal("rho_gp", mu=np.log(rho_gp_guess), 
                                      sigma=self.rho_gp_sigma,
                                      testval=rho_gp_guess)
                ## non-periodic stochastically-driven, damped harmonic oscillator
                kernel = terms.SHOTerm(sigma=sigma_gp, rho=rho_gp, Q=0.25)

                ## trying Matern 3/2
                #kernel = terms.Matern32Term(sigma=sigma_gp,rho=rho_gp)           
                gp = GaussianProcess(kernel, t=x[mask], yerr=sigma_lc,quiet=True)

                light_curves_no_GP = pm.Deterministic("lc_no_GP",light_curves_semifinal)
                resid = y[mask] - light_curves_no_GP[mask]

                gp.marginal("gp", observed=resid)
                
                light_curves_final = pm.Deterministic("lc_final",light_curves_no_GP  + gp.predict(resid,t=x))
            else:
                light_curves_final = pm.Deterministic("lc_final",light_curves_semifinal)
                pm.Normal("obs", mu=light_curves_final[mask], sd=sigma_lc, observed=y[mask])
    
        resultDict = {}
        resultDict['model'] = model
        resultDict['x'] = x
        resultDict['y'] = y
        resultDict['yerr'] = yerr
        resultDict['mask'] = mask
        resultDict['specModel'] = specModel
        resultDict['specInfo'] = specInfo
        #resultDict['gp'] = gp
        resultDict['waveName'] = waveName
        
        return resultDict



    def get_wavebin(self,nbins=None,waveBinNum=0,
                    forceRecalculate=None):
        
        if nbins == None:
            nbins = self.nbins
        if forceRecalculate is None:
            if waveBinNum == 0:
                recalculate = True
            else:
                recalculate = False
        else:
            recalculate = forceRecalculate
        
        t1, t2 = self.spec.get_wavebin_series(nbins=nbins,
                                              binStarts=self.wbin_starts,
                                              binEnds=self.wbin_ends,recalculate=recalculate)
    
        x1 = np.ascontiguousarray(t1['Time'],dtype=np.float64)
        waveName = t1.colnames[1+waveBinNum]
        y1 = np.ascontiguousarray(t1[waveName] * 1000.) ## convert to ppt
        yerr1 = np.ascontiguousarray(t2[t2.colnames[1+waveBinNum]] * 1000.) ## convert to ppt



        return x1,y1,yerr1,waveName

    def build_model_spec(self, mask=None,start=None,waveBinNum=0,nbins=None,
                         forceRecalculate=None):
        
        if nbins == None:
            nbins = self.nbins
        
        ## broadband fit
        broadband = ascii.read(self.broadband_fit_file)
        x1, y1, yerr1, waveName1 = self.get_wavebin(nbins=nbins,waveBinNum=waveBinNum,
                                                    forceRecalculate=forceRecalculate)
    
        if mask is None:
            mask = np.ones(len(x1), dtype=bool)
        
        specInfo = {}
        specInfo['broadband'] = broadband
        specInfo['x'] = x1
        if self.differentialMode == True:
            specInfo['y'] = y1/self.y
            specInfo['yerr'] = yerr1/self.y
            specInfo['BBmxap_soln'] = self.bbRes['map_soln']
        else:
            specInfo['y'] = y1
            specInfo['yerr'] = yerr1
        specInfo['waveName'] = waveName1
        waveStart, waveEnd, waveMid, dispIndicesTable = self.lookup_waveBins(nbins=nbins)
        specInfo['waveMid'] = waveMid[waveBinNum]

        resultDict = self.build_model(specInfo=specInfo)
        resultDict['specInfo'] = specInfo
        
    
        #resultDict['gp'] = gp
    
        return resultDict

    def find_mxap(self,resultDict,start=None,recalculate=False):
        if start is None:
            #start = model.test_point
            start = resultDict['model'].test_point
        
        model = resultDict['model']

        with model:
            # t0 = model.t0
            # ror = model.ror
            # incl = model.incl
            # u_star = model.u_star
            # mean = model.mean
            
            if self.fitSigma == 'fit':
                allvars = model.vars
                initial_vars = []
                for oneVar in allvars:
                    if oneVar.name == 'sigma_lc':
                        pass
                    else:
                        initial_vars.append(oneVar)
                map_soln = pmx.optimize(start=start,vars=initial_vars)
                map_soln = pmx.optimize(start=map_soln)
            else:
                map_soln = pmx.optimize(start=start)
                
            #map_soln = pmx.optimize(start=start, vars=[t0])
            # map_soln = pmx.optimize(start=start, vars=[ror])
            # # map_soln = pmx.optimize(start=map_soln, vars=[incl])
            # #
            #map_soln = pmx.optimize(start=map_soln, vars=[u_star])
            # map_soln = pmx.optimize(start=map_soln, vars=[ror])
            # # map_soln = pmx.optimize(start=map_soln, vars=[incl])
            # #
            # # map_soln = pmx.optimize(start=map_soln, vars=[mean])
            #map_soln = pmx.optimize(start=map_soln)
        
        resultDict['map_soln'] = map_soln
        
        return resultDict
    
    def update_mask(self,mxapDict):
        """
        Update the mask to exclude outliers. Use the MAP solution
        """
        resid = mxapDict['y'] - mxapDict['map_soln']['lc_final']
        newMask = (np.abs(resid) < self.sigReject * mxapDict['yerr'])
        ## make sure it doesn't add in any new points that were masked out at the start 
        self.mask = newMask & self.startMask
        
    def find_mxap_with_clipping(self,modelDict,iterations=2,
                                extraDescrip='',recalculate=False):
        outName = 'mxap_{}{}.npz'.format(self.descrip,extraDescrip)
        mxapDir = os.path.join('fit_mxap',self.descrip)
        make_sure_of_path(mxapDir)
        mxapPath = os.path.join(mxapDir,outName)

        if (os.path.exists(mxapPath) == True) & (recalculate == False):
            npzfiles  = np.load(mxapPath)#,allow_pickle=True)
            mxapSoln = dict(npzfiles)
            mxapDict = modelDict
            mxapDict['map_soln'] = mxapSoln
        else:

            for oneIter in np.arange(iterations):
                if oneIter > 0:
                    ## only update the mask after the first run
                    self.update_mask(mxapDict)
                    specInfo = deepcopy(modelDict['specInfo'])
                    modelDict = self.build_model(specInfo=specInfo)
                    
                mxapDict = self.find_mxap(modelDict)
            np.savez(mxapPath,**mxapDict['map_soln'])
        
        return mxapDict
    
    
    def save_mxap_lc(self,mxapDict=None):
        if mxapDict is None:
            modelDict = self.build_model()
            mxapDict = self.find_mxap_with_clipping(modelDict)
        t = Table()

        t['Time'] = self.x
        t['Flux'] = self.y
        t['Flux Err'] = self.yerr
        
        
        if 'light_curves' in mxapDict['map_soln']:
            light_curves = mxapDict['map_soln']['light_curves']
            if len(light_curves.shape) > 1:
                lc = light_curves[:,0]
            else:
                lc = light_curves[:]
            t['model']= lc
        
        if 'lc_final' in mxapDict['map_soln']:
            t['lc_final'] = mxapDict['map_soln']['lc_final']
        
        results_to_keep = ['mean', 'a', 'incl', 'period', 't0',
                           'omega', 'u_star', 'ror', 'depth', 'ecc']
        
        
        t.meta = {}
        for one_param in results_to_keep:
            if one_param in mxapDict['map_soln']:
                val = mxapDict['map_soln'][one_param]
                if len(val.shape) == 0:
                    t.meta[one_param] = float(val)
                else:
                    t.meta[one_param] = list(val)
                
        outName = 'mxap_lc_{}.ecsv'.format(self.descrip)
        outPath = os.path.join("fit_results","mxap_soln","mxap_lc",outName)
        direct1 = os.path.split(outPath)[0]
        if os.path.exists(direct1) == False:
            os.makedirs(direct1)
        t.write(outPath,overwrite=True)
    
    def find_spectrum(self,nbins=None,doInference=False,
                      startInd=0):
        """
        Find the spectrum by looping through the wavelength bins

        Parameters
        ----------
        nbins: int
            Number of wavelength bins to loop through
        doInference: bool
            Find the posterior distribution? (Otherwise it finds the mxap solution only)

        startInd: int
            Starting index for the loop (for continuing a previously stopped run)
        """
        
        if nbins == None:
            nbins = self.nbins
        
        bin_arr = np.arange(startInd,nbins)
        depth_list = []
        ## make sure the wavelength bins are established
        t1, t2 = self.spec.get_wavebin_series(nbins=nbins,
                                              binStarts=self.wbin_starts,
                                              binEnds=self.wbin_ends,recalculate=True)
        tnoise = self.spec.print_noise_wavebin(nbins=nbins,recalculate=False)
        waveList = tnoise['Wave (mid px)']
        for oneBin in bin_arr:
            if self.differentialMode == True:
                self.bbRes = self.run_all_broadband()
            
            modelDict1 = self.build_model_spec(waveBinNum=oneBin,nbins=nbins)
            waveName = "{}_nbins_{}".format(waveList[oneBin],nbins)
            extraDescrip="_{}".format(waveName)
            if doInference == True:
                resultDict = self.find_posterior(modelDict1,extraDescrip=extraDescrip)
                
                t = self.print_es_summary(resultDict,broadband=False,
                                     waveName=waveName)
            else:
                x1, y1, yerr1, waveName1 = self.get_wavebin(nbins=nbins,waveBinNum=oneBin)
                mapDict = self.find_mxap_with_clipping(modelDict1,
                                                       extraDescrip=extraDescrip)
                depth_list.append(mapDict['map_soln']['depth'])
                resultDict = mapDict
            
            self.plot_test_point(resultDict,extraDescrip='_{}'.format(waveName))
        return depth_list, waveList
    
    
    def collect_bb_fits(self):
        """
        Collect the broadband lightcurves, errors, and model fits
        """
        modelDict = self.build_model()
        resultDict = self.find_posterior(modelDict)
        t = Table()
        t['x'] = resultDict['x']
        t['y'] = resultDict['y']
        t['yerr'] = resultDict['yerr']
        t['mask'] = resultDict['mask']
        lc_dict_sys = self.get_lc_stats(resultDict,lc_var_name='lc_final')
        t['model_sys'] = lc_dict_sys['median']
        lc_dict_astroph = self.get_lc_stats(resultDict,lc_var_name='light_curves')
        t['model_astroph'] = (np.array(lc_dict_astroph['median']) + 1.) * 1000.
        
        return t
    
    def collect_all_lc_and_fits(self,nbins=None,
                                recalculate=False,
                                astrophParameterization='v02'):
        """
        Collect all spectroscopic lightcurves, errors and model fits
                                
        Parameters
        ----------
        nbins: int
            Number of wavelength bins
        recalculate: bool
            Recalculate the lightcurves? If False, will look for a previously
                                saved FITS file
        """
        if nbins == None:
            nbins = self.nbins
        
        outDir = os.path.join('fit_results',self.descrip)
        if os.path.exists(outDir) == False:
            os.makedirs(outDir)
        outName = 'lc_2Dfits_{}_nbins_{}.fits'.format(self.descrip,nbins)
        outPath = os.path.join(outDir,outName)
        
        if (os.path.exists(outPath) == False) | (recalculate == True):
        
            bin_arr = np.arange(nbins)
            ror_list = []
            tnoise = self.spec.print_noise_wavebin(nbins=nbins,recalculate=False)
            waveList_midpx = tnoise['Wave (mid px)']
            x_list = []
            y_list = []
            yerr_list = []
            model_list_astroph = []
            model_list_sys = []
            mask_list = []
            if self.differentialMode == True:
                self.bbRes = self.run_all_broadband()
        
            for oneBin in tqdm.tqdm(bin_arr):
                modelDict1 = self.build_model_spec(waveBinNum=oneBin,nbins=nbins,
                                                   forceRecalculate=False)
                modelDict1['map_soln'] = 'placeholder'
                waveName = "{}_nbins_{}".format(waveList_midpx[oneBin],nbins)
                resultDict = self.find_posterior(modelDict1,extraDescrip="_{}".format(waveName))
                x_list.append(modelDict1['x'])
                y_list.append(modelDict1['y'])
                yerr_list.append(modelDict1['yerr'])
            
                ## Get the fitted lightcurves
                lc_dict_sys = self.get_lc_stats(resultDict,lc_var_name='lc_final')
                model_list_sys.append(lc_dict_sys['median'])
                lc_dict_astroph = self.get_lc_stats(resultDict,lc_var_name='light_curves')
                model_list_astroph.append(lc_dict_astroph['median'])
                
                mask_list.append(modelDict1['mask'])
        
            result = {}
            waveStart, waveEnd, waveMid, dispIndicesTable = self.lookup_waveBins(nbins=nbins)    
            result['wave'] = waveMid
            result['x'] = np.array(x_list)
            result['y'] = np.array(y_list)
            result['yerr'] = np.array(yerr_list)
            result['mask'] = np.array(mask_list)
            if astrophParameterization == 'old-legacy':
                result['model_astroph'] = (np.array(model_list_astroph) + 1.) * 1000.
            else:
                result['model_astroph'] = np.array(model_list_astroph)
            result['model_sys'] = np.array(model_list_sys)
            
            self.save_lc_fits2D(result,outPath)
        else:
            HDUList = fits.open(outPath)
            result = {}
            result['wave'] = HDUList['WAVE'].data
            result['x'] = HDUList['TIME'].data
            result['y'] = HDUList['FLUX'].data
            result['yerr'] = HDUList['FLUXERR'].data
            result['mask'] = HDUList['MASK'].data
            result['model_astroph'] = HDUList['MODASTROPH'].data
            result['model_sys'] = HDUList['MODSYS'].data
            HDUList.close()
        
        return result
    
    def save_lc_fits2D(self,res,outPath):
        nbins = len(res['wave'])

        primHDU = fits.PrimaryHDU(None)
        primHDU.header['DESCRIP'] = (self.descrip,'lc fit desscription')
        primHDU.header['NWAVEB'] = (nbins,'Number of wavelength bins used')
    
        timeHDU = fits.ImageHDU(res['x'])
        timeHDU.header['BUNIT'] = ('JD','Time in Julian Date, days')
        timeHDU.name = 'TIME'
    
        fluxHDU = fits.ImageHDU(res['y'])
        fluxHDU.header['BUNIT'] = ('ppt','Normalized Flux Units')
        fluxHDU.name = 'FLUX'
    
        ferrHDU = fits.ImageHDU(res['yerr'])
        ferrHDU.header['BUNIT'] = ('ppt','Normalized Flux Error Units')
        ferrHDU.name = 'FLUXERR'
        
        maskHDU = fits.ImageHDU(np.array(res['mask'],dtype=int))
        maskHDU.header['BUNIT'] = ('bool','0 for discard outlier, 1 for keep')
        maskHDU.name = 'MASK'
        
        modAHDU = fits.ImageHDU(res['model_astroph'])
        modAHDU.header['BUNIT'] = ('ppt','Normalized Flux Units')
        modAHDU.name = 'MODASTROPH'
        
        modSysHDU = fits.ImageHDU(res['model_sys'])
        modSysHDU.header['BUNIT'] = ('ppt','Normalized Flux Units')
        modSysHDU.name = 'MODSYS'
        

        
        waveHDU = fits.ImageHDU(res['wave'])
        waveHDU.header['BUNIT'] = ('microns','units of wavelength')
        waveHDU.name = 'WAVE'
    
        outHDU = fits.HDUList([primHDU,timeHDU,fluxHDU,ferrHDU,
                              maskHDU,
                              modAHDU,modSysHDU,waveHDU])
    
        print("Saving file to {}".format(outPath))
        outHDU.writeto(outPath,overwrite=True)
    
    def lookup_waveBins(self,nbins):
        """
        Look up the wavelength bin starts, middles and ends
        
        """
                ## Use the edges of the pixels for the bin starts/ends
        HDUList = fits.open(self.spec.wavebin_specFile(nbins=nbins))
        dispIndicesTable = Table(HDUList['DISP INDICES'].data)
        waveStart=np.round(self.spec.wavecal(dispIndicesTable['Bin Start']-0.5),5)
        waveEnd=np.round(self.spec.wavecal(dispIndicesTable['Bin End']-0.5),5)

        if self.correctedBinCenters == True:
            waveMid = (waveStart + waveEnd)/2.
            midPxWave = self.spec.wavecal(dispIndicesTable['Bin Middle']-0.5)
            dispIndicesTable['Wave mid px'] = np.round(midPxWave,5)
        else:
            waveMid = self.spec.wavecal(dispIndicesTable['Bin Middle']-0.5)

        return waveStart, waveEnd, waveMid, dispIndicesTable

    def collect_spectrum(self,nbins=None,doInference=False,
                         redoWaveBinCheck=True,
                         gatherAll=False):
        if nbins == None:
            nbins = self.nbins
        
        bin_arr = np.arange(nbins)
        
        tnoise = self.spec.print_noise_wavebin(nbins=nbins)
        waveList = tnoise['Wave (mid px)']

        if gatherAll == True:
            spec_dict = {}
            spec_dict_err = {}
        else:
            depth, depth_err = [], []
        
        if self.fit_t0_spec == True:
            t0, t0_err = [], []
        for ind,oneBin in enumerate(bin_arr):
            fileName = "{}_wave_{}_nbins_{}_fit.csv".format(self.descrip,waveList[oneBin],nbins)
            dat = ascii.read(os.path.join('fit_results',self.descrip,fileName))
            if gatherAll == True:
                if ind == 0:
                    all_vars = dat['var name']
                    for oneVar in all_vars:
                        spec_dict[oneVar] = []
                        spec_dict_err[oneVar] = []
                for oneVar in all_vars:
                    spec_dict[oneVar].append(get_from_t(dat,oneVar,'mean'))
                    spec_dict_err[oneVar].append(get_from_t(dat,oneVar,'std'))
            else:
                depth.append(get_from_t(dat,'depth','mean'))
                depth_err.append(get_from_t(dat,'depth','std'))
            
            if self.fit_t0_spec == True:
                t0.append(get_from_t(dat,'t0','mean'))
                t0_err.append(get_from_t(dat,'t0','std'))


        ## make sure the wavelength bins are established
        t1, t2 = self.spec.get_wavebin_series(nbins=nbins,
                                              binStarts=self.wbin_starts,
                                              binEnds=self.wbin_ends,
                                              recalculate=redoWaveBinCheck)
        print("Making sure wavelength bins are established from bin parameters")

        waveStart, waveEnd, waveMid, dispIndicesTable = self.lookup_waveBins(nbins=nbins)

        t = Table()
        t['wave start'] = waveStart
        ## average wavelength is not the same as wavelenght of avg pixel
        t['wave mid'] = waveMid
        t['wave end'] = waveEnd
        t['wave width'] = np.round(t['wave end'] - t['wave start'],4)

        if gatherAll == True:
            for oneVar in all_vars:
                t[oneVar] = spec_dict[oneVar]
                t[oneVar+' err'] = spec_dict_err[oneVar]
        else:
            t['depth'] = depth
            t['depth err'] = depth_err
            if self.fit_t0_spec == True:
                t['t0'] = t0
                t['t0 err'] = t0_err
        t['pxbin start'] = dispIndicesTable['Bin Start']
        t['pxbin end'] = dispIndicesTable['Bin End']
        t['pxbin mid'] = dispIndicesTable['Bin Middle']
        ## average wavelength is not the same as wavelenght of avg pixel
        t['pxbin mid wv'] = self.spec.wavecal(t['pxbin mid'])
        outName = self.specFileName
        t.write(outName,overwrite=True)
        return t    
    
    def plot_spec(self,closeFig=True,redoWaveBinCheck=True):
        """
        Plot the spectrum from all the sampling runs
        closeFig: bool
            Close the figure?
        redoWaveBinCheck: bool
            Re-calculate the bin wavelength starts and ends?
        """
        t = self.collect_spectrum(redoWaveBinCheck=redoWaveBinCheck)
        fig, ax = plt.subplots()
        ax.errorbar(t['wave mid'],t['depth'] * 1e6,yerr=t['depth err'] * 1e6)
        ax.set_ylabel('Depth (ppm)')
        ax.set_xlabel('Wavelength ($\mu$m)')
        
        make_sure_of_path('plots/spectra_pdf')
        make_sure_of_path('plots/spectra_png')
        
        fig.savefig('plots/spectra_pdf/spectrum_simple_{}.pdf'.format(self.descrip),
                    bbox_inches='tight',facecolor='white')
        fig.savefig('plots/spectra_png/spectrum_simple_{}.png'.format(self.descrip),
                    bbox_inches='tight',facecolor='white',dpi=250)
        if closeFig == True:
            plt.close(fig)

    def plot_param_spec(self,param,redoWaveBinCheck=True):
        """
        Plot the spectrum of different parameters in the fitting
        (e.g.) linear slope
        """
        make_sure_of_path('plots/param_spec_png')
        fig, ax = plt.subplots()
        t = self.collect_spectrum(gatherAll=True,redoWaveBinCheck=redoWaveBinCheck)
        xval = t['wave mid']
        yval = t[param]
        yval_err = t[param + ' err']
        ax.errorbar(xval,yval,yerr=yval_err)
        ax.set_xlabel("Wavelength (ppm)")
        ax.set_ylabel(param)
        specPath = 'plots/param_spec_png/{}_spec_{}.png'.format(param,self.descrip)
        print("Saving file to {}".format(specPath))
        fig.savefig(specPath,
                    bbox_inches='tight',dpi=150,facecolor='white')


    def plot_test_point(self,modelDict,extraDescrip='',yLim=[None,None],
                        yLim_resid=[None,None],redoWaveBinCheck=True,
                        showDetrended=False):
        """
        Check the guess or the maximum a priori lightcurve
    
        modelDict: dict with 'model'
            Dictionary for model. If a 'map_soln' is found, it will be plotted
        extraDescrip: str
            Extra description to be saved in plot name
        yLim: 2 element list of floats or None
            Y limits for lightcurve plot
        """
        if 'map_soln' in modelDict:
            testpt = modelDict['map_soln']
            map_soln = True
            
            light_curve = modelDict['map_soln']['lc_final']
        else:
            #testpt = modelDict['model'].test_point
            with modelDict['model']:
                 #light_curve = pmx.eval_in_model(self.sys.flux(t=self.x))
                 light_curve = pmx.eval_in_model(modelDict['model']['lc_final'])
            
            map_soln = False
        # # Orbit model
#
#         if 'ecc' in testpt:
#
#             ecc = testpt['ecc']
#             omega=testpt['omega']
#         elif 'ecc_interval__' in testpt:
#             print("Could not find plain eccentricity. Assuming self.ecc value")
#             ecc = self.ecc[0]
#             omega = self.omega[0]
#         else:
#             ecc = 0.0
#             omega = 90.0
#
#         orbit = xo.orbits.KeplerianOrbit(
#             period=testpt['period'],
#             a=testpt['a'],
#             incl=testpt['incl'] * np.pi/180.,
#             t0=testpt['t0'],
#             ecc=ecc,
#             omega=omega * np.pi/180.,
#         )
#
#         if 'u_star' in testpt:
#             u_star = testpt['u_star']
#         elif 'u_star_quadlimbdark__' in testpt:
#             u_star = testpt['u_star_quadlimbdark__']
#         else:
#             ## For spectra, u_star is fixed at the broadband value for now
#             if self.u_lit == None:
#                 broadband = ascii.read('fit_results/broadband_fit_{}.csv'.format(self.descrip))
#                 u_star = [get_from_t(broadband,'u_star__0','mean'),
#                           get_from_t(broadband,'u_star__0','mean')]
#             else:
#                 u_star = self.u_lit
#
#         light_curves_obj = xo.LimbDarkLightCurve(u_star)
#         ror = np.exp(testpt['ror_log__'])
#         light_curves_var = light_curves_obj.get_light_curve(orbit=orbit, r=ror,
#                                                             t=self.x, texp=self.texp)
#
#         light_curve =  (np.sum(light_curves_var.eval(),axis=-1) + 1.) * testpt['mean']
#         ## the mean converts to parts per thousand
#
#         logp_est = -0.5 * np.sum((light_curve - self.y)**2/self.yerr**2)
#         print('Logp (rough) = {}'.format(logp_est))
        ## the old way
        dataMask = modelDict['mask']

        if showDetrended == True:
            fig, (ax,ax1,ax2) = plt.subplots(3,sharex=True)
        else:
            fig, (ax,ax2) = plt.subplots(2,sharex=True)

        ax.errorbar(modelDict['x'],modelDict['y'],yerr=modelDict['yerr'],
                    fmt='.',zorder=0,color='red')
        ax.errorbar(modelDict['x'][dataMask],modelDict['y'][dataMask],
                    yerr=modelDict['yerr'][dataMask],
                    fmt='.',zorder=1,label='Raw F')
        ax.plot(modelDict['x'],light_curve,linewidth=3,zorder=2)

        if showDetrended == True:
            if map_soln == True:
                astroph_lc = modelDict['map_soln']['light_curves']
                sysModel = light_curve / astroph_lc
                yDetrend = modelDict['y'] / sysModel * 1e3
                yDetrend_err = modelDict['yerr'] / sysModel * 1e3
                ax1.errorbar(modelDict['x'][dataMask],yDetrend[dataMask],
                             yerr=yDetrend_err[dataMask],fmt='.',
                             zorder=1,label='Detrended')
                ax1.plot(modelDict['x'],astroph_lc * 1e3,linewidth=3,zorder=2)
                ax1.legend()
                ax1.set_ylabel("Flux (ppt)")
                ax.legend()

        resid = modelDict['y'] - light_curve
        if self.nbins_resid is None:
            resid_alpha = 1.0
        else:
            resid_alpha = 0.7
        
        ax2.errorbar(modelDict['x'][dataMask],resid[dataMask],
                     yerr=modelDict['yerr'][dataMask],fmt='.',
                     alpha=resid_alpha)
        
        if self.nbins_resid is None:
            pass
        else:
            x_bin, y_bin, y_bin_err = phot_pipeline.do_binning(modelDict['x'][dataMask],
                                                            resid[dataMask],nBin=self.nbins_resid)
            if self.equalize_bin_err == True:
                y_bin_err = np.ones_like(y_bin_err) * np.nanmedian(y_bin_err)
            
            plt.errorbar(x_bin,y_bin,fmt='o')
        
        ax.set_ylabel("Flux (ppt)")
        ax.set_ylim(yLim[0],yLim[1])
    
        ax2.set_ylabel("Residual (ppt)")
        ax2.set_xlabel("Time (JD)")
        ax2.set_ylim(yLim_resid[0],yLim_resid[1])

        if showDetrended == True:
            extraDescrip = '_w_detrend' + extraDescrip

        combined_descrip = 'mapsoln_{}_{}_{}'.format(map_soln,self.descrip,extraDescrip)
        lc_path = os.path.join('plots','lc_plots',self.descrip)
        if os.path.exists(lc_path) == False:
            os.makedirs(lc_path)
        for suffix in ['.pdf','.png']:
            outName='lc_plot_{}{}'.format(combined_descrip,suffix)
            fig.savefig(os.path.join(lc_path,outName),
                        dpi=200,facecolor='white',
                        bbox_inches='tight')
        
    
        fig, ax = plt.subplots()
        ax.hist(resid,bins=np.linspace(-0.7,0.7,32),density=True)
        
        ax.set_ylabel("Relative Frequency")
        ax.set_xlabel("Flux Resid (ppt)")
        stat, critv, siglevel = stats.anderson(resid)
        ax.set_title('A$^2$={:.3f}'.format(stat))
        try:
            fig.savefig('plots/resid_histos/lc_resids_{}.pdf'.format(combined_descrip))
        except FileNotFoundError as error:
            os.makedirs('plots/resid_histos/')
            fig.savefig('plots/resid_histos/lc_resids_{}.pdf'.format(combined_descrip))
        plt.close(fig)

    def plot_lc_from_mxap(self,mapDict,mask=None):
        """
        Plot the lightcurve from the Maxim A Priori solution
        """
        if mask is None:
            mask = np.ones_like(self.x,dtype=bool)
        
        plt.plot(self.x,self.y,'o')
        plt.plot(self.x[mask],mapDict['map_soln']['lc_final'],
                 color='black',zorder=2)
        plt.show()

    def find_posterior(self,modelDict=None,extraDescrip='',recalculate=False):
        """
        Find the Posterior distribution with the No U Turns pymc3 sampler
        """
        if modelDict is None:
            modelDict = self.build_model()
    
        if 'map_soln' not in modelDict:
            resultDict = self.find_mxap_with_clipping(modelDict,
                                                      extraDescrip=extraDescrip)
        else:
            resultDict = modelDict
    
        model0 = resultDict['model']
    
        outDir = 'fit_traces/fits_{}{}'.format(self.descrip,extraDescrip)
        all_chains = glob.glob(os.path.join(outDir,'*'))
        
        if (len(all_chains) < self.nchains) | (recalculate == True):
            if os.path.exists(outDir) == False:
                os.makedirs(outDir)
            
            with model0: 
                trace = pmx.sample( 
                    tune=3000, 
                    draws=3000, 
                    start=resultDict['map_soln'], 
                    cores=self.cores, 
                    chains=self.nchains, 
                    init=self.pymc3_init, 
                    target_accept=0.9, 
                )
            pm.save_trace(trace, directory =outDir, overwrite = True)
        else:
            with model0:
                trace = pm.load_trace(directory=outDir)
    
        resultDict['trace'] = trace
    

    
    
    
        return resultDict

    def print_es_summary(self,resultDict,broadband=True,waveName=None,
                         saveChisq=True):
        ## set up two variable arrays to grab posteriors from pandas_dataframe
        ## varnames are the variable names used by pymc3
        ## varList is what they will look like in the pandas dataframe
        ## sometimes the pandas dataframe has multiple values for one
        ## variable. For example, u_star goes to u_star__0 and u_star__1
        if broadband == True:
            varnames = ['mean','a','incl','t0','ror','depth','period']
            varList = ['mean','a','incl','t0','ror','depth','period']

        else:
            varnames = ['mean','a','incl','t0','ror','depth']
            varList = deepcopy(varnames)
        
        if self.eclipseGeometry == 'PhaseCurve':
            additional_varNames = ['phaseOffset','phase_amp',
                                   'e_depth']
            additional_varList = ['phaseOffset','phase_amp',
                                  'e_depth']
            if self.phaseCurveFormulation == 'standard':
                additional_varNames.append('nightsideDepth')
                additional_varNames.append('planetFullamp')
                additional_varList.append('nightsideDepth')
                additional_varList.append('planetFullamp')
            
            for ind,oneVar in enumerate(additional_varNames):
                varnames.append(oneVar)
                varList.append(additional_varList[ind])

        if self.u_lit == None:
            varnames.append('u_star')
            varList.append('u_star__0')
            varnames.append('u_star')
            varList.append('u_star__1')

            if (self.fixLDu1 == True) & (broadband==False):
                ## I named the variable here manually
                varnames.append('u_star__1')
                varList.append('u_star__1')
        
        if (self.ecc != 'zero') & (broadband == True):
            varnames.append('ecc')
            varnames.append('omega')
            varList.append('ecc')
            varList.append('omega')
        
        if (self.fitSigma == 'fit'):
            varnames.append('sigma_lc')
            varList.append('sigma_lc')
        
        
        if 'poly' in self.trendType:
            for oneCoeff in np.arange(self.poly_ord):
                varnames.append('poly_coeff')
                varList.append('poly_coeff__{}'.format(oneCoeff))
        
        if 'fpah' in self.trendType:
            varnames.append('fpahCoeff')
            varList.append('fpahCoeff')
        
        if 'refpix' in self.trendType:
            varnames.append('refpixCoeff')
            varList.append('refpixCoeff')

        if self.offsetMask is None:
            pass
        else:
            for oneStep in np.arange(self.nSteps - 1):
                varnames.append('stepOffsets')
                varList.append('stepOffsets__{}'.format(oneStep))

        if (self.expStart == True):
            varnames.append('exp_tau')
            varnames.append('exp_amp')
            varList.append('exp_tau')
            varList.append('exp_amp')
        elif (self.expStart == 'double'):
            varnames.extend(['exp_tau1','exp_tau2','exp_amp1','exp_amp2'])
            varList.extend(['exp_tau1,','exp_tau2','exp_amp1','exp_amp2'])
        
        ## check if variable is in posterior and only keep the ones that are
        available_vars = []
        available_varList = []
        for ind,checkVar in enumerate(varnames):
            if checkVar in resultDict['trace'].varnames:
                ## make sure it isn't already added to the list (such as exp_tau and exp_tau_log__ appearing twice)
                ## But we want some things that appear twice like u_star__0 and u_star__1
                if varList[ind] not in available_varList:
                    available_varList.append(varList[ind])
                if checkVar not in available_vars:
                    available_vars.append(checkVar)
            
        samples = pm.trace_to_dataframe(resultDict['trace'], varnames=available_vars)
        
        names, means, stds = [], [], []
        for oneVar in available_varList:
            mean=np.mean(samples[oneVar])
            std1 = np.std(samples[oneVar])
            #print("Var {},mean={}, std={}".format(oneVar,mean,std1))
            means.append(mean)
            stds.append(std1)
        
        ## Save q1 and q2 if available
        if ('u_star__0' in samples) & ('u_star__1' in samples):
            q1 = (samples['u_star__0'] + samples['u_star__1'])**2
            q2 = samples['u_star__0'] / (2. * (samples['u_star__0'] + samples['u_star__1']))
            available_varList.append('q1')
            means.append(np.mean(q1))
            stds.append(np.std(q1))
            available_varList.append('q2')
            means.append(np.mean(q2))
            stds.append(np.std(q2))

        if saveChisq == True:
            ## save the likelihood info
            mask1 = resultDict['mask']
            pts = np.sum(mask1)
            resid = (resultDict['y'][mask1] - resultDict['map_soln']['lc_final'][mask1]) 
            chisq_full = np.sum(resid**2 / resultDict['yerr'][mask1]**2)
            nparams = len(resultDict['model'].basic_RVs)
            dof = pts - nparams
            norm_chisq = chisq_full / dof
            BIC = nparams * np.log(pts) + 2. * chisq_full

            available_varList.append('norm chisq')
            means.append(norm_chisq)
            available_varList.append('BIC')
            means.append(BIC)
            stds.append(np.nan)
            stds.append(np.nan)


        t = Table()
        t['var name'] = available_varList
        t['mean'] = means
        t['std'] = stds
        
        if os.path.exists('fit_results') == False:
            os.makedirs('fit_results')
        if broadband == True:
            t.write(self.broadband_fit_file,overwrite=True)
        else:
            outDir = os.path.join('fit_results',self.descrip)
            if os.path.exists(outDir) == False:
                os.makedirs(outDir)
            t.write(os.path.join(outDir,'{}_wave_{}_fit.csv'.format(self.descrip,waveName)),overwrite=True)
        return t
    
    def lookup_result(self,resultDict,name):
        """
        Look up a value and an uncertainty from a result dictionary
        """
        row = resultDict['var name']==name
        mean = resultDict['mean'][row][0]
        stdev = resultDict['std'][row][0]
        return mean, stdev

    def get_tmid(self,resultDict):
        """
        Calculate the transit mid-point and uncertainty
        """
        bestPeriod,bestPeriod_err = self.lookup_result(resultDict,'period')
        bestT0, bestT0_err = self.lookup_result(resultDict,'t0')
        if self.eclipseGeometry == 'Transit':
            refEpoch = bestT0
        else:
            if self.ecc == 'zero':
                refEpoch = bestT0 + 0.5 * bestPeriod
            else:
                raise NotImplementedError
        ntrans = np.round((np.median(self.x) - refEpoch)/bestPeriod)
        
        tmid = ntrans * bestPeriod + refEpoch
        tmid_err = np.sqrt((ntrans * bestPeriod_err)**2 + bestT0_err**2)

        return tmid, tmid_err
    # def plot_trace(resultDict):
    #     _ = pm.traceplot(resultDict['trace'], var_names=["mean","logr_pl","b","u_star","rho_gp"])
    #     plt.savefig('plots/pymc3/traceplot.pdf')
    #     plt.close()
    ## placeholder text for printing info on coefficients
    # def lookup(res,var):
    #     pt = res['var name'] == var
    #     return res[pt]
    #
    # def print_coefficients(posteriorResult,self):
    #     if self.
    #     mult_coeff = Table()
    #     varNames = ['A','B','C']
    #     mean_flux = lookup(posteriorResult,'mean')
    #     A = mean_flux['mean'][0]
    #     A_err = mean_flux['std'][0]
    #
    #     B_table = lookup(posteriorResult,'poly_coeff__0')
    #     B = B_table['mean'][0] * A
    #     B_err = np.sqrt((B_table['std'][0] * A)**2 + (A_err * B_table['mean'][0])**2)
    #
    #     C_table = lookup(posteriorResult,'poly_coeff__1')
    #     C = C_table['mean'][0] * A
    #     C_err = np.sqrt((C_table['std'][0] * A)**2 + (A_err * C_table['mean'][0])**2)
    #
    #     mult_coeff['var name'] = varNames
    #     mult_coeff['Mean Val (ppt)'] = np.round([A,B,C],3)
    #     mult_coeff['Err (ppt)'] = np.round([A_err,B_err,C_err],3)
    #
    #     time_length = (np.max(em4.x) - np.min(em4.x)) * 24.
    #     mult_coeff['Absolute Units'] = ['(ppt/hr)^{}'.format(i) for i in range(3)]
    #     mult_coeff['Val in Absolute Units'] = np.round(time_length**np.arange(3) * mult_coeff['Mean Val (ppt)'],3)
    #
    #     return mult_coeff
    
    def get_one_wave_result_dict(self,oneBin,nbins=None):
        if nbins == None:
            nbins = self.nbins

        tnoise = self.spec.print_noise_wavebin(nbins=nbins,recalculate=False)
        waveList_midpx = tnoise['Wave (mid px)']

        modelDict1 = self.build_model_spec(waveBinNum=oneBin,nbins=nbins,
                                                forceRecalculate=False)
        modelDict1['map_soln'] = 'placeholder'
        waveName = "{}_nbins_{}".format(waveList_midpx[oneBin],nbins)
        resultDict = self.find_posterior(modelDict1,extraDescrip="_{}".format(waveName))
        return resultDict

    def corner_plot(self,resultDict,compact=True,
                    re_param_u=True,
                    truths=None,range=None):
        if self.re_param_u == True:
            if re_param_u == True:
                limb_dark = "u_star_quadlimbdark__"
            else:
                limb_dark = 'u_star'
        else:
            limb_dark = 'u_star'
    
        if compact == True:
            varnames = ["depth"]
            outName = 'cornerplot_compact.pdf'
        else:
            
            potential_varnames = ["depth","a","incl","t0"]
            varnames = []
            for oneVarName in potential_varnames:
                if oneVarName in resultDict['trace'].varnames:
                    varnames.append(oneVarName)
            if 'poly_coeff' in resultDict['model'].test_point:
                varnames.append('poly_coeff')
            outName = 'cornerplot_full.pdf'
        
        if (self.u_lit == None) & (self.eclipseGeometry == 'Transit'):
            varnames.append(limb_dark)
        
        samples = pm.trace_to_dataframe(resultDict['trace'], varnames=varnames)
        
        #_ = corner.corner(samples)
        # truths = [0.00699764850849,None, None]
        #,range=[(0.0068,0.00740),(-2.35,-1.90),(-4.5,2.0)])
        _ = corner.corner(samples,truths=None)
        try:
            plt.savefig('plots/corners/{}_{}'.format(self.descrip,outName))
        except FileNotFoundError as error:
            os.makedirs('plots/corners/')
            plt.savefig('plots/corners/{}_{}'.format(self.descrip,outName))
        plt.close()
    
    # def planet_upper_size_limit(resultDict):
    #     samples = pm.trace_to_dataframe(resultDict['trace'], varnames=["mean","logr_pl","b"])
    #     ## This should be in solar radii
    #     logr_limit = np.percentile(samples['logr_pl'],95)
    #     rpl_upper = 10**logr_limit * u.Rsun
    #     print("Rp/R* upper = {}".format(rpl_upper/r_star))
    #     print("Rpl upper = {}".format(rpl_upper.to(u.km)))
    #     print("Rpl upper = {}".format(rpl_upper.to(u.Rearth)))
    #
    #     return rpl_upper.to(u.Rearth)
    #
    # def planet_r_histogram(resultDict):
    #     samples = pm.trace_to_dataframe(resultDict['trace'], varnames=["logr_pl"])
    #     logr_pl = np.array(samples['logr_pl'])
    #     logr_re = np.log10((10**logr_pl * u.Rsun).to(u.Rearth).value)
    #
    #     fig, ax = plt.subplots()
    #     _ = ax.hist(logr_re)
    #
    #     perc_thresh = 95
    #     max_logr_re = np.percentile(logr_re,perc_thresh)
    #     max_r_re = 10**max_logr_re
    #     print('Max R {} (R_e) at {}%'.format(max_r_re,perc_thresh))
    #
    #     labelText = '{}%, {:.2f} R$_\oplus$'.format(perc_thresh,max_r_re)
    #     ax.text(max_logr_re + 0.1,450,labelText,rotation=90,fontsize=14)
    #
    #     ax.axvline(max_logr_re,color='red')
    #     ax.set_ylabel("Relative Frequency")
    #     ax.set_xlabel("Log$_{10}$(R / R$_\oplus$)")
    #
    #     fig.savefig('plots/pymc3/{}_r_posterior_re.pdf'.format(self.descrip))

    def get_pm_summary(self,resultDict):
        pm_summary = pm.summary(resultDict['trace'])
    
        return pm_summary

    def get_lc_stats(self,resultDict,lc_var_name = 'lc_final'):
        with resultDict['model']:
            xdataset = arviz.convert_to_dataset(resultDict['trace'])
        
        if lc_var_name == 'light_curves':
            lcDataset = xdataset[lc_var_name]
            if len(lcDataset.shape) == 4:
                lcCube = lcDataset[:,:,:,0]
            else:
                lcCube = lcDataset
        else:
            lcCube = xdataset[lc_var_name]
        final_lc2D = flatten_chains(lcCube)
        lc_dict = {}
        lc_dict['median'] = np.median(final_lc2D,axis=0)
        lc_dict['lim'] = np.percentile(final_lc2D,axis=0,q=[2.5,97.5])
        
        return lc_dict
    
    def plot_lc_stats_from_xdata(self,lc_dict=None,resultDict=None):
        if lc_dict is None:
            resultDict = self.find_posterior()
            lc_dict = self.get_lc_stats(resultDict)
        
        plt.plot(self.x,self.y,'o')
        plt.fill_between(self.x,lc_dict['lim'][0], lc_dict['lim'][1],
                         color='black',alpha=0.6,zorder=2)
        plt.show()

    def plot_lc_from_summary(self,pm_summary,rho_gp=None,r_p=None,includeGP=True):
        lc= calc_lightcurve_from_summary(pm_summary,rho_gp=None,r_p=r_p,
                                         includeGP=includeGP)
        #plt.errorbar(x,y,yerr=self.yerr)
        plt.plot(self.x,self.y,'o')
        plt.plot(self.x,lc.eval())
        plt.show()
    
    def get_hdi_array(self,resultDict,prob=0.95):
        res = pm.hpd(resultDict['trace'],var_names=['final_lc'])
        return np.array(res['final_lc'])
    
    def gather_spec_lc_stats(self):
        """
        Gather the lightcurve statistics from spectroscopic lightcurves
        """
        lcfit1 = self.collect_all_lc_and_fits()
        resid1 = lcfit1['y'] - lcfit1['model_sys']
        stdevArr1 = np.nanstd(resid1,axis=1)
        resid1[lcfit1['mask']==0] = np.nan
        stdevArr2 = np.nanstd(resid1,axis=1)
        yerrArr = np.nanmedian(lcfit1['yerr'],axis=1)
        res = {}
        res['wave'] = lcfit1['wave']
        res['yerr'] = yerrArr
        res['unmasked std'] = stdevArr1
        res['std'] = stdevArr2
        if self.fitSigma == 'fit':
            spAll = self.collect_spectrum(gatherAll=True)
            res['fit sigma'] = spAll['sigma_lc']
        else:
            res['fit sigma'] = np.zeros(len(res['wave'])) * np.nan
        return res

    def plot_lc_distribution(self,lc_hdi,pm_summary=None,rpl_upper=None):
        fig, ax = plt.subplots()
        #ax.errorbar(x,y,yerr=self.yerr)
        ax.plot(self.x,self.y,'o')
    
        ax.errorbar(dat_bin['Time'],dat_bin['Flux'] * 1000.,yerr=dat_bin['Flux Err'] * 1000.)
    
        y1, y2 = lc_hdi[:,0], lc_hdi[:,1]
        ax.fill_between(self.x,y1=y1,y2=y2,color='orange',alpha=0.4,label='GP Posterior')
        outFile = 'plots/pymc3/lc_distribution.pdf'
        print("Saving plot to {}".format(outFile))
    
        ax.set_xlabel("Time (BJD)")
        ax.set_ylabel("Flux (p.p.t.)")
    
        if pm_summary is not None:
            ## upper limit from Sanchis-Ojeda et al. 2015
            refVal1 = 2.5 * u.Rearth
            r_p_so = (refVal1).to(u.Rsun).value
            lc= calc_lightcurve_from_summary(pm_summary,rho_gp=None,r_p=r_p_so,
                                             includeGP=False)
            ax.plot(self.x,lc.eval(),label='{}'.format(refVal1))
        
            if rpl_upper is not None:
                r_p_plugin = (rpl_upper).to(u.Rsun).value
                lc= calc_lightcurve_from_summary(pm_summary,rho_gp=None,r_p=r_p_plugin,
                                                 includeGP=False)
                ax.plot(self.x,lc.eval(),label='{:.2f}'.format(rpl_upper))
        
            ax.legend()
    
        fig.savefig(outFile)
        plt.close(fig)

    def run_all_broadband(self,recalculate=False):
        modelDict = self.build_model()
        mapDict = self.find_mxap_with_clipping(modelDict,
                                               recalculate=recalculate)
        self.save_mxap_lc(mapDict)
        for showDetrended in [True,False]:
            self.plot_test_point(mapDict,showDetrended=showDetrended)
        postDict = self.find_posterior(mapDict,recalculate=recalculate)
        self.print_es_summary(postDict)
        self.corner_plot(postDict,compact=False)
        
        return postDict

    def run_all(self,plotMxapFirst=True):
        self.run_all_broadband()
        if plotMxapFirst == True:
            ror_list, wavelist = self.find_spectrum(doInference=False) ## plot max a priori sol
        ror_list, wavelist = self.find_spectrum(doInference=True)
        self.plot_spec()
    
    
def get_from_t(tab,var,val):
    """
    Get a value from a table
    """
    pts = tab['var name'] == var
    assert np.sum(pts) == 1
    return tab[pts][val][0]

def make_sure_of_path(path):
    """
    Check if a path exists
    If not, make it
    """
    if os.path.exists(path) == False:
        os.makedirs(path)
    
# def get_limits_and_plot():
#     resultDict = find_posterior()
#     plot_trace(resultDict)
#     self.corner_plot(resultDict)
#     rpl_upper = planet_upper_size_limit(resultDict)
#     lc_hdi = get_hdi_array(resultDict)
#     pm_summary = get_pm_summary(resultDict)
#
#     plot_lc_distribution(lc_hdi,pm_summary,rpl_upper=rpl_upper)
#     return resultDict, pm_summary, lc_hdi

if __name__ == "__main__":
    mod = exo_model()
    mod.run_all()


spec_to_comp1 = 'fit_results/mirage_023_grismr_newspec_ncdhas/spectrum_mirage_023_grismr_newspec_ncdhas.csv'
spec_to_comp2 = 'fit_results/mirage_023_grismr_newspec_ncdhas_fixLD/spectrum_mirage_023_grismr_newspec_ncdhas_fixLD.csv'

def compare_spectra(specList=[spec_to_comp1,spec_to_comp2],
                    labelList=['Free LD','Fixed LD at truth'],
                    showPlot=False,waveOffset=0.0):
    """
    Compare 2 or more spectra from different analysis techniques
    
    Parameters
    ----------
    specList: list of str
        List of paths to .csv files for spectra
    labelList: list of str
        List of labels for the spectra. Must be same length as specList
    showPlot: bool
        If True, eender the plot with show(). If False, save to file.
    """
    fig, (ax0,ax1) = plt.subplots(2,sharex=True)
    
    specStorage = []
    for ind,oneSpec in enumerate(specList):
        dat = ascii.read(oneSpec)
        dat.sort('wave')
        if ind == 1:
            dat['wave'] = dat['wave'] + waveOffset
        specStorage.append(dat)
        ax0.errorbar(dat['wave'],dat['depth'] * 1e6,yerr=dat['depth err'] * 1e6,label=labelList[ind])
    ax0.set_ylabel("Depth (ppm)")
    ax0.legend()
    ax1.set_xlabel("Wavelength ($\mu$m)")
    
    for ind1,dat in enumerate(specStorage):
        wave1 = dat['wave']
        depth1 = dat['depth']
        err1 = dat['depth err']
        for ind2,dat2 in enumerate(specStorage):
            if ind2 > ind1:
                wave2 = dat2['wave']
                depth2 = dat2['depth']
                err2 = dat2['depth err']
                comb_err = np.sqrt(err1**2 + err2**2)
                diff = depth2 - depth1
                
                ax1.errorbar(wave1,diff * 1e6,yerr=comb_err * 1e6)
    ax1.set_ylabel("Difference (ppm)")
    ax1.axhline(0.0,color='black',linestyle='dotted')
    
    if showPlot == True:
        fig.show()
    else:
        fig.savefig('plots/comparison_spectra/comparison_spectra.pdf')

def calc_starry_model(time,a=8.0,period=3.0,
                      u_star=[0.1,0.1],
                      ror=0.1,
                      amp=0.0,t0=2459870.0,
                      ecc=0.0,omega=90.0,
                      incl=90.0,texp=5.0):
    """ 
    Calculate a starry lightcurve model with a simple set of parameters
    Note that it is very slow to calculate the first time

    Parameters
    ----------
    time: numpy array
        Time, in days
    a: float
        Semi-major axis in stellar radius units, (e.g. 8.0)
    period: float
        Orbital period, days (e.g. 3.0)
    u_star: list of floats
        Limb darkening polynomial law (eg. [0.1, 0.1])
    ror: float
        planet radius to star radius ratio
    amp: float
        planet eclipse depth (or amplitude of map)
        only will affect eclipse depth, but will re-normalize
        if applied to transit (e.g. 1e-3)
    t0: float
        epoch of transit center (e.g. 2459870.0)
    ecc: float
        orbital eccentricity (e.g. 0.0)
    omega: float
        argument of pericenter (degrees) (e.g. 90.0)
    inc: float
        inclination (degrees) (e.g. 90.0)
    texp: float
        exposure time (seconds) (e.g. 5.0)
    """
    Msys = unitless_prefac * a**3 / period**2
    m_star = Msys ## assume it's all star for now
    
    star_map = starry.Map(udeg=len(u_star))
    star_map[1:] = u_star
    star = starry.Primary(star_map,m=m_star,r=1.0)
    
    planet = starry.kepler.Secondary(starry.Map(amp=amp),
                                        m=0.0,
                                        r=ror,
                                        porb=period,
                                        prot=period,
                                        t0=t0,
                                        ecc=ecc,
                                        omega=omega,
                                        inc=incl)
    sys = starry.System(star,planet,texp=texp/(24. * 3600.))
    light_curve = sys.flux(t=time)
    return light_curve.eval()

def flatten_chains(trace3D):
    """
    Flatten points in the chain to give distributions across all chains
    
    Inputs
    ----------
    trace3D: 3D or other numpy array
        The 3D array with the Python shape nchains x npoints x nvariables
    
    Outputs
    --------
    trac2D: 2D numpy array
        The 2D array with the Python shape n_allpoints x nvariables
    """
    nchains, npoints, nvariables = trace3D.shape
    
    trace2D = np.reshape(np.array(trace3D),[nchains * npoints,nvariables])
    return trace2D

def do_wavebins(flux2D,err2D,binSize):
    """ Calculate the theoretical an measured noise in lightcurves """

    shape2D = flux2D.shape
    nints, nwav = shape2D[0], shape2D[1]
    
    nbins = int(np.floor(nwav / binSize ))

    flux_binned2D = np.zeros([nints,nbins])
    err_binned2D = np.zeros_like(flux_binned2D)
    for one_bin in np.arange(nbins):
        ind_st = one_bin * binSize
        ind_end = ind_st + binSize
        flux_binned2D[:,one_bin] = np.nansum(flux2D[:,ind_st:ind_end],axis=1) / binSize
        err_binned2D[:,one_bin] = np.sqrt(np.nansum(err2D[:,ind_st:ind_end]**2,axis=1)) / binSize
    
    resultDict = {}
    resultDict['nbins'] = nbins
    resultDict['flux_binned2D'] = flux_binned2D
    resultDict['std_binned'] = np.nanstd(flux_binned2D,axis=0)
    resultDict['theo_median'] = np.nanmedian(err_binned2D)#,axis=0)
    return resultDict

def calculate_nightside_spec(sp_all):
    """
    Calculate the nightside spectrum from a phase curve
    """
    nightsideF = sp_all['e_depth'] * (1.0 - 2. * sp_all['phase_amp'])
    # fracDepthErr = sp_all['e_depth err'] / sp_all['e_depth']
    # fracAmpErr = sp_all['phase_amp err'] / sp_all['phase_amp']
    # product_err = sp_all['e_depth'] * sp_all['phase_amp'] * np.sqrt(fracDepthErr**2 + fracAmpErr**2)
    fullAmp, fullAmp_err = calculate_full_amp(sp_all)
    nightsideF_var = sp_all['e_depth err']**2 + fullAmp_err**2
    nightsideF_err = np.sqrt(nightsideF_var)
    
    return nightsideF,nightsideF_err

def calculate_full_amp(sp_all):
    """
    Calculate the full ampltiude = 2. * fractional amp * eclipse depth
    """
    fullAmp = sp_all['e_depth'] * 2. * sp_all['phase_amp']
    fracDepthErr = sp_all['e_depth err'] / sp_all['e_depth']
    fracAmpErr = sp_all['phase_amp err'] / sp_all['phase_amp']
    product_err = sp_all['e_depth'] * sp_all['phase_amp'] * np.sqrt(fracDepthErr**2 + fracAmpErr**2)
    fullAmp_err = product_err * 2.
    return fullAmp, fullAmp_err


def allanvar_wave(flux2D,err2D,showFloor=None,
                  binMax=2**12):
    """
    Calculate the allan variance as a function of wavelength bin
    """
    nwav = flux2D.shape[1]
    binpts_arr = 2**np.arange(int(np.log2(binMax)))
    
    measured_list = []
    theoretical_list = []
    nbins_list = []

    for one_binSize in binpts_arr:
        resultDict = do_wavebins(flux2D,err2D,one_binSize)
        nbins_list.append(resultDict['nbins'])
        measured_list.append(resultDict['std_binned'])
        theoretical_list.append(resultDict['theo_median'])

    fig, ax = plt.subplots()
    theo_arr = np.array(theoretical_list) * 1e6
    theo_eval = theo_arr[0] / np.sqrt(binpts_arr/binpts_arr[0])
    #ax.loglog(binpts_arr,theo_arr,label="Ideal 1/$\sqrt{N}$")
    ax.loglog(binpts_arr,theo_eval,label="Ideal 1/$\sqrt{N}$")
    if showFloor is None:
        pass
    else:
        theo_with_floor = np.sqrt(theo_arr**2 + (showFloor)**2)
        floorLabel = "With {} ppm extra noise".format(showFloor)
    #ax.loglog(binpts_arr,theo_with_floor,label=floorLabel)
    
    for ind,one_binSize in enumerate(binpts_arr):
        ax.plot(np.ones(nbins_list[ind]) * one_binSize,
                measured_list[ind] * 1e6,'o')
    ax.legend()
    ax.set_xlabel("N Wavebins (px)")
    ax.set_ylabel("Error (ppm)")
    
    #ax.axhline(0.20)
    
    outDir = 'plots/allan_variance_wavebin'
    if os.path.exists(outDir) == False:
        os.makedirs(outDir)
    outPath = os.path.join(outDir,'all_var_wavebin.png')
    print("Writing plot to {}".format(outPath))
    fig.savefig(outPath,
                dpi=150,bbox_inches='tight')
    #print(binpts_arr)

overlapWaveRangeToCheck = [3.88,4.02]
def calc_mean_in_overlap(sp,waveRange=overlapWaveRangeToCheck):
    pts_overlap = ((sp['wave mid'] > overlapWaveRangeToCheck[0] ) & 
               (sp['wave mid'] < overlapWaveRangeToCheck[1]))
    meanInOverlap = np.mean(sp['depth'][pts_overlap] * 1e6)
    errInOverlap = np.median(sp['depth err'][pts_overlap] * 1e6) / np.sqrt(np.sum(pts_overlap))
    return meanInOverlap, errInOverlap

def compare_overlap(sp1,sp2,waveRange=overlapWaveRangeToCheck):
    mean1, err1 = calc_mean_in_overlap(sp1)
    mean2, err2 = calc_mean_in_overlap(sp2)
    diff = mean1 - mean2
    diff_err = np.sqrt(err1**2 + err2**2)
    return diff, diff_err

def get_irac_bp(iracFilter='3.6'):
    if iracFilter == '3.6':
        bp = ascii.read(os.path.join(os.environ['TEL_DATA'],'irac/irac_36_ang.txt'))
    elif iracFilter == '4.5':
        bp = ascii.read(os.path.join(os.environ['TEL_DATA'],'irac/irac_45_ang.txt'))
    else:
        msg = "IRAC {} either isn't implemented or does not exist".format(iracFilter)
        raise NotImplementedError(msg)

    bp_interp = interp1d(bp['col1']/1e4,bp['col2'],bounds_error=False,
                         fill_value=0.0)
    return bp_interp

def calc_syn_irac(wave,depth,depth_err,iracFilter):
    """ 
    Calculate the synthetic photometry from IRAC
    """
    bp = get_irac_bp(iracFilter=iracFilter)
    syn_depth, syn_depth_err = calc_syn_phot(wave,depth,depth_err,bp)
    return syn_depth, syn_depth_err

def calc_syn_phot(wave,depth,depth_err,bp):
    bp_i = bp(wave)
    normVal = np.nansum(bp_i)
    syn_depth = np.nansum(bp_i * depth) / normVal
    syn_depth_err = np.sqrt(np.nansum(bp_i**2 * depth_err**2)) / normVal
    return syn_depth, syn_depth_err

def syn_phot_table(specList):
    """
    Takes a list of spectra and calculates IRAC photometry
    """
    filterList = []
    synDepthList = []
    synDepthErrList = []
    for oneSpec in specList:
        minWave = np.min(oneSpec['wave mid'])
        if minWave < 3.5:
            iracFilterSynth = '3.6'
        elif minWave < 5.0:
            iracFilterSynth = '4.5'
        else:
            msg = "IRAC {} either isn't implemented or does not exist".format(iracFilterSynth)
            raise NotImplementedError(msg)

        filterList.append(iracFilterSynth)
        res = calc_syn_irac(oneSpec['wave mid'],
                            oneSpec['depth'],
                            oneSpec['depth err'],
                            iracFilter=iracFilterSynth)
        synDepthList.append(res[0])
        synDepthErrList.append(res[1])
    t_syn = Table()
    t_syn['Filter'] = filterList
    t_syn['Depth'] = np.round(np.array(synDepthList) * 1e6,1)
    t_syn['Depth err'] = np.round(np.array(synDepthErrList) * 1e6,1)
    return t_syn

def calc_wav_sp(spList):
    """
    Calculate the Weighted Average of Spectra
    Assuming they all of the same wavelength bins
    """
    nSpec = len(spList)
    
    nWaves = len(spList[0]['wave mid'])
    allDepths = np.zeros([nWaves,nSpec])
    weights = np.zeros_like(allDepths)
    for spInd in np.arange(nSpec):
        allDepths[:,spInd] = spList[spInd]['depth']
        weights[:,spInd] = 1./spList[spInd]['depth err']**2
    
    norm = np.sum(weights,axis=1)
    avgVal = np.sum(allDepths * weights,axis=1)/norm
    avgErr = 1./np.sqrt(norm)

    spOut = Table()
    spOut['wave mid'] = spList[0]['wave mid']
    spOut['wave width'] = spList[0]['wave width']
    spOut['depth'] = avgVal
    spOut['depth err'] = avgErr

    return spOut

if __name__ == "__main__":
    freeze_support()

