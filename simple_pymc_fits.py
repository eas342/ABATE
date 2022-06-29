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

class exo_model(object):
    
    def __init__(self,paramPath=default_paramPath,descrip=default_descrip,t0_lit=defaultT0,
                 recalculateTshirt=True,period_lit=default_period,inc_lit=default_inc,
                 a_lit=default_a,u_lit=None,pipeType='spec',
                 ecc=default_ecc,omega=default_omega,
                 ld_law=default_ld,sigReject=10,
                 starry_ld_degree=default_starry_ld_deg,
                 cores=2,nchains=2,
                 fitSigma=None,
                 nbins=20,
                 oot_start=0,oot_end=100):
        #paramPath = 'parameters/spec_params/jwst/sim_mirage_007_grismc/spec_mirage_007_p015_full_emp_cov_weights_nchdas_mmm.yaml'
        #paramPath = 'parameters/spec_params/jwst/sim_mirage_007_grismc/spec_mirage_007_p016_full_emp_cov_weights_ncdhas_ppm.yaml'
        # paramPath = 'parameters/spec_params/jwst/sim_mirage_009_grismr_8grp/spec_mirage_009_p012_full_cov_highRN_nchdas_ppm_refpix.yaml'
        # descrip = 'grismr_001'
    
        self.paramPath = paramPath
        self.descrip = descrip
        self.t0_lit = t0_lit
        self.inc_lit = inc_lit
        self.a_lit = a_lit
        self.u_lit = u_lit
        self.ecc = ecc
        self.omega = omega
        self.ld_law = ld_law
        self.cores = cores
        self.nchains = nchains
        if ld_law == 'quadratic':
            self.starry_ld_degree = None
        else:
            self.starry_ld_degree = starry_ld_degree
        # paramPath = 'parameters/spec_params/jwst/sim_mirage_009_grismr_8grp/spec_mirage_009_p014_ncdhas_mpm_skip_xsub.yaml'
        # descrip = 'grismr_003_no_xsub'

        #t0_lit = 2459492.8347039, 0.000035
        ## Something was wrong with those, so I'm just putting the center I got from one fit
        
        self.nbins = nbins ## wavelength bins for spectroscopy
        self.paramFile = os.path.join(tshirtDir,paramPath)
        
        if pipeType == 'phot':
            self.phot = phot_pipeline.phot(self.paramFile)
            t1, t2 = self.phot.get_tSeries()
            timeKey = 'Time (JD)'
            ## normalize
            normVal = np.nanmedian(t1[t1.colnames[1]])
            t1[t1.colnames[1]] = t1[t1.colnames[1]] / normVal
            t2[t2.colnames[1]] = t2[t2.colnames[1]] / normVal
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

        self.texp = np.min(np.diff(self.x))

        self.mask = np.ones(len(self.x), dtype=bool)
        self.sigReject = sigReject
        
        # Orbital parameters for the planet, Sanchis-ojeda 2015
        self.period_lit = period_lit

        #t0_lit = 2459491.323591, 0.000035 #GRISMC has different one for some reason
        #t0_lit = 2459560.576056, 0.000035 # GRISMR 
        #t0_lit = 2459560.576056, 0.000035
        #t0_lit = (t0_lit[0] + 0.5705 * period_lit[0], 0.05)
        
        np.random.seed(42) 
    
        # Allow a re-calculation of yerr in case it is under-estimated
        self.fitSigma = fitSigma
        # self.useOOTforError = useOOTforError
        self.oot_start = oot_start
        self.oot_end = oot_end
    
    
    def check_phase(self):
        phase = (self.x - self.t0_lit[0]) / self.period_lit[0]
        return phase

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
            if self.u_lit == None:
                if self.ld_law == 'quadratic':
                    u_star = xo.QuadLimbDark("u_star",testval=[0.71,0.1])
                else:
                    # u_star = pm.Lognormal("u_star",mu=np.log(0.1), sigma=0.5,
#                                             shape=(default_starry_ld_deg,))
                    ld_start = np.zeros(self.starry_ld_degree) + 0.1
                    #ld_start[0] = 0.1
                    testVal = ld_start
                    u_star = pm.Normal("u_star",mu=ld_start,testval=ld_start,
                                       sigma=2.0,
                                       shape=(self.starry_ld_degree,))
            else:
                u_star = self.u_lit
        
            if specInfo == None:
                a = pm.Normal("a",mu=self.a_lit[0],sigma=self.a_lit[1],testval=self.a_lit[0])
                incl = pm.Normal("incl",mu=self.inc_lit[0],sigma=self.inc_lit[1],testval=self.inc_lit[0])
                # BoundedNormal = pm.Bound(pm.Normal, lower=0, upper=3)
                # m_star = BoundedNormal(
                #     "m_star", mu=M_star[0], sd=M_star[1]
                # )
                # r_star = BoundedNormal(
                #     "r_star", mu=R_star[0], sd=R_star[1]
                # )
                
                period = pm.Normal("period",mu=self.period_lit[0],sd=self.period_lit[1])
                
                t0 = pm.Normal("t0", mu=self.t0_lit[0], sd=self.t0_lit[1])
                #t0 = t0_lit[0]
                
                ror = pm.Lognormal("ror", mu=np.log(0.0822), sigma=0.5)
                
                #ror = pm.Deterministic("ror", tt.pow(10,logr_pl) / R_star[0])#r_star)
                #b = xo.distributions.ImpactParameter("b", ror=ror)
                ecc_from_broadband = False
                
                x, y, yerr = self.x, self.y, self.yerr
                specModel = False
                waveName = 'broadband'
            else:
                broadband = specInfo['broadband']
                a = get_from_t(broadband,'a','mean')
                # a = pm.Normal("a",mu=get_from_t(broadband,'a','mean'),
#                               sigma=get_from_t(broadband,'a','std'))
                incl = get_from_t(broadband,'incl','mean')
                # incl = pm.Normal("incl",mu=get_from_t(broadband,'incl','mean'),
#                                  sigma=get_from_t(broadband,'incl','std'))
                
                period = get_from_t(broadband,'period','mean')
                #period = pm.Normal("period",mu=self.period_lit[0],sd=self.period_lit[1])
                
                t0 = get_from_t(broadband,'t0','mean')
                # t0 = pm.Normal("t0", mu=get_from_t(broadband,'t0','mean'),
                #                  sigma=get_from_t(broadband,'t0','std'))
                #t0 = t0_lit[0]
                
                mean_r = get_from_t(broadband,'ror','mean')
                
                ror = pm.Lognormal("ror", mu=np.log(mean_r), sigma=0.3)
                
                if 'ecc' in broadband['var name']:
                    ecc_from_broadband = True
                    ecc_from_broadband_val = get_from_t(broadband,'ecc','mean')
                    # ecc_from_broadband_val = (get_from_t(broadband,'ecc','mean'),
                    #                           get_from_t(broadband,'ecc','std'))
                    
                else:
                    ecc_from_broadband = False
                    ecc_from_broadband_val = (0, 0.02)
                
                    
                
                x, y, yerr = specInfo['x'], specInfo['y'], specInfo['yerr']
                specModel = True
                waveName = specInfo['waveName']
            
            # if self.useOOTforError == True:
#                 yerr = np.std(y[self.oot_start:self.oot_end]) * np.ones_like(yerr)
            if self.fitSigma == 'fit':
                sigma_lc = pm.Lognormal("sigma_lc", mu=np.log(np.median(yerr)), sigma=0.5)
            elif self.fitSigma == 'oot':
                sigma_lc = np.std(y[self.oot_start:self.oot_end])
                
            else:
                sigma_lc = yerr[mask]
            
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
                ecc = pm.TruncatedNormal("ecc",mu=self.ecc[0],sigma=self.ecc[1],lower=0.0,upper=1.0)
                omega = pm.Normal("omega",mu=self.omega[0],sigma=self.omega[1])
            # xo.eccentricity.kipping13("ecc_prior", fixed=True, observed=ecc)
        
            # ecc = 0.0
            # omega = np.pi/2.0
            # xo.eccentricity.kipping13("ecc_prior", fixed=True, observed=ecc)
            
            if self.ld_law == 'quadratic':
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
                light_curves_var = light_curves_obj.get_light_curve(orbit=orbit, r=ror,
                                                                    t=x, texp=self.texp)
        
                # Compute the model light curve
                light_curves = pm.Deterministic("light_curves",light_curves_var)
        
                light_curve = (tt.sum(light_curves, axis=-1) + 1.) * mean
            
                
            else:
                
                ## Astropy units don't see to carry through, so ...
                ## Calculate Kepler's 3rd law with a prefactor
                prefac = (2. * np.pi / (1.0 * u.day))**2 * (1.0 * u.Rsun)**3 / const.G
                unitless_prefac = prefac.to(u.Msun).value
                
                Msys = unitless_prefac * a**3 / period**2
                m_star = Msys ## assume it's all star for now
                
                star_map = starry.Map(udeg=self.starry_ld_degree)
                star_map[1:] = u_star
                star = starry.Primary(star_map,m=m_star,r=1.0)
                
                planet = starry.kepler.Secondary(starry.Map(amp=0),
                                                 m=0.0,
                                                 r=ror,
                                                 porb=period,
                                                 prot=period,
                                                 t0=t0,
                                                 ecc=ecc,
                                                 omega=omega,
                                                 inc=incl)
                sys = starry.System(star,planet,texp=self.texp)
                light_curve = sys.flux(t=x) * mean
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
            
            light_curves_final = pm.Deterministic("lc_final",light_curve)
            
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
            # ## the Gaussian process error should be larger given all the ground-based systematics
            # sigma_gp = pm.Lognormal("sigma_gp", mu=np.log(np.std(self.y[mask]) * 5.), sigma=0.5)
            #
            # tau_gp = pm.Lognormal("tau_gp",mu=np.log(1e-2), sigma=0.5)
            #
            # kernel = terms.SHOTerm(sigma=sigma_gp, rho=rho_gp, tau=tau_gp)
            # ## trying Matern 3/2
            # #kernel = terms.Matern32Term(sigma=sigma_gp,rho=rho_gp)
            #
            # gp = GaussianProcess(kernel, t=x[mask], yerr=sigma_lc,quiet=True)
            # gp.marginal("gp", observed=resid)
            # #gp_pred = pm.Deterministic("gp_pred", gp.predict(resid))
            # final_lc = pm.Deterministic("final_lc",light_curve + gp.predict(resid))
            #
            # # Fit for the maximum a posteriori parameters, I've found that I can get
            # # a better solution by trying different combinations of parameters in turn
        
        
            pm.Normal("obs", mu=light_curve[mask], sd=sigma_lc, observed=y[mask])
        
            #pdb.set_trace()

    
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



    def get_wavebin(self,nbins=None,waveBinNum=0):
        
        if nbins == None:
            nbins = self.nbins
        
        t1, t2 = self.spec.get_wavebin_series(nbins=nbins)
    
        x1 = np.ascontiguousarray(t1['Time'],dtype=np.float64)
        waveName = t1.colnames[1+waveBinNum]
        y1 = np.ascontiguousarray(t1[waveName] * 1000.) ## convert to ppt
        yerr1 = np.ascontiguousarray(t2[t2.colnames[1+waveBinNum]] * 1000.) ## convert to ppt
    
        return x1,y1,yerr1,waveName

    def build_model_spec(self, mask=None,start=None,waveBinNum=0,nbins=None):
        
        if nbins == None:
            nbins = self.nbins
        
        ## broadband fit
        broadband = ascii.read('fit_results/broadband_fit_{}.csv'.format(self.descrip))
        x1, y1, yerr1, waveName1 = self.get_wavebin(nbins=nbins,waveBinNum=waveBinNum)
    
        if mask is None:
            mask = np.ones(len(x1), dtype=bool)
        
        specInfo = {}
        specInfo['broadband'] = broadband
        specInfo['x'] = x1
        specInfo['y'] = y1
        specInfo['yerr'] = yerr1
        specInfo['waveName'] = waveName1
        
        resultDict = self.build_model(specInfo=specInfo)
        
        
    
        #resultDict['gp'] = gp
    
        return resultDict

    def find_mxap(self,resultDict,start=None):
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
        self.mask = np.abs(resid) < self.sigReject * mxapDict['yerr']
        
    def find_mxap_with_clipping(self,modelDict,iterations=2):
        for oneIter in np.arange(iterations):
            if oneIter > 0:
                ## only update the mask after the first run
                self.update_mask(mxapDict)
                specInfo = deepcopy(modelDict['specInfo'])
                modelDict = self.build_model(specInfo=specInfo)
                
            mxapDict = self.find_mxap(modelDict)
        
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
            lc = mxapDict['map_soln']['light_curves'][:,0]
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
    
    def find_spectrum(self,nbins=None,doInference=False):
        if nbins == None:
            nbins = self.nbins
        
        bin_arr = np.arange(nbins)
        ror_list = []
        tnoise = self.spec.print_noise_wavebin(nbins=nbins)
        waveList = tnoise['Wave (mid)']
        for oneBin in bin_arr:
            modelDict1 = self.build_model_spec(waveBinNum=oneBin)
            waveName = "{}_nbins_{}".format(waveList[oneBin],nbins)
            if doInference == True:
                resultDict = self.find_posterior(modelDict1,extraDescrip="_{}".format(waveName))
            
                t = self.print_es_summary(resultDict,broadband=False,
                                     waveName=waveName)
            else:
                x1, y1, yerr1, waveName1 = self.get_wavebin(nbins=nbins,waveBinNum=oneBin)
                mapDict = self.find_mxap_with_clipping(modelDict1)
                self.plot_test_point(mapDict,extraDescrip='_{}'.format(waveName))
                ror_list.append(mapDict['map_soln']['ror'])
        return ror_list, waveList
    

    def collect_spectrum(self,nbins=None,doInference=False):
        if nbins == None:
            nbins = self.nbins
        
        bin_arr = np.arange(nbins)
        depth, depth_err = [], []
        tnoise = self.spec.print_noise_wavebin(nbins=nbins)
        waveList = tnoise['Wave (mid)']
        for oneBin in bin_arr:
            fileName = "{}_wave_{}_nbins_{}_fit.csv".format(self.descrip,waveList[oneBin],nbins)
            dat = ascii.read(os.path.join('fit_results',self.descrip,fileName))
            depth.append(get_from_t(dat,'depth','mean'))
            depth_err.append(get_from_t(dat,'depth','std'))
    
        t = Table()
        t['wave'] = waveList
        t['depth'] = depth
        t['depth err'] = depth_err
        outName = os.path.join('fit_results',self.descrip,'spectrum_{}.csv'.format(self.descrip))
        t.write(outName,overwrite=True)
        return t    
    
    def plot_spec(self):
        t = self.collect_spectrum()
        fig, ax = plt.subplots()
        ax.errorbar(t['wave'],t['depth'] * 1e6,yerr=t['depth err'] * 1e6)
        ax.set_ylabel('Depth (ppm)')
        ax.set_xlabel('Wavelength ($\mu$m) (needs updating)')
        fig.savefig('plots/spectra_pdf/spectrum_simple_{}.pdf'.format(self.descrip),bbox_inches='tight')
        fig.savefig('plots/spectra_png/spectrum_simple_{}.png'.format(self.descrip),bbox_inches='tight',dpi=250)
        plt.close(fig)

    def plot_test_point(self,modelDict,extraDescrip='',yLim=[None,None],
                        yLim_resid=[None,None]):
        """
        Check the guess lightcurve
    
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
                 light_curve = pmx.eval_in_model(self.sys.flux(t=mod.x))
            
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

    
        fig, (ax,ax2) = plt.subplots(2,sharex=True)
        ax.errorbar(modelDict['x'],modelDict['y'],yerr=modelDict['yerr'],
                    fmt='.',zorder=0,color='red')
        ax.errorbar(modelDict['x'][self.mask],modelDict['y'][self.mask],
                    yerr=modelDict['yerr'][self.mask],
                    fmt='.',zorder=1)
        ax.plot(self.x,light_curve,linewidth=3,zorder=2)
    
        resid = modelDict['y'] - light_curve
        ax2.errorbar(modelDict['x'][self.mask],resid[self.mask],
                     yerr=modelDict['yerr'][self.mask])
    
        ax.set_ylabel("Flux (ppt)")
        ax.set_ylim(yLim[0],yLim[1])
    
        ax2.set_ylabel("Residual (ppt)")
        ax2.set_xlabel("Time (JD)")
        ax2.set_ylim(yLim_resid[0],yLim_resid[1])
    
        combined_descrip = 'mapsoln_{}_{}_{}'.format(map_soln,self.descrip,extraDescrip)
        lc_path = os.path.join('plots','lc_plots',self.descrip)
        if os.path.exists(lc_path) == False:
            os.makedirs(lc_path)
        fig.savefig(os.path.join(lc_path,'lc_plot_{}.pdf'.format(combined_descrip)))
        fig.savefig(os.path.join(lc_path,'lc_plot_{}.png'.format(combined_descrip)))
    
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

    def find_posterior(self,modelDict=None,extraDescrip=''):
    
        if modelDict is None:
            modelDict = self.build_model()
    
        if 'map_soln' not in modelDict:
            resultDict = self.find_mxap_with_clipping(modelDict)
        else:
            resultDict = modelDict
    
        model0 = resultDict['model']
    
        outDir = 'fit_traces/fits_{}{}'.format(self.descrip,extraDescrip)
        all_chains = glob.glob(os.path.join(outDir,'*'))
        
        if len(all_chains) < self.nchains:
            if os.path.exists(outDir) == False:
                os.makedirs(outDir)
            
            with model0: 
                trace = pm.sample( 
                    tune=3000, 
                    draws=3000, 
                    start=resultDict['map_soln'], 
                    cores=self.cores, 
                    chains=self.nchains, 
                    init="adapt_full", 
                    target_accept=0.9, 
                )
            pm.save_trace(trace, directory =outDir, overwrite = True)
        else:
            with model0:
                trace = pm.load_trace(directory=outDir)
    
        resultDict['trace'] = trace
    

    
    
    
        return resultDict

    def print_es_summary(self,resultDict,broadband=True,waveName=None):
    
        if broadband == True:
            varnames = ['mean','a','incl','t0','ror','depth','period']
            varList = ['mean','a','incl','t0','ror','depth','period']
            if self.u_lit == None:
                varnames.append('u_star')
                varList.append('u_star__0')
                varList.append('u_star__1')
        else:
            varnames = ['mean','a','incl','t0','ror','depth']
            varList = varnames
        
        if (self.ecc != 'zero') & (broadband == True):
            varnames.append('ecc')
            varnames.append('omega')
            varList.append('ecc')
            varList.append('omega')
        
        
        ## check if variable is in posterior and only keep the ones that are
        available_vars = []
        available_varList = []
        for ind,checkVar in enumerate(varnames):
            if checkVar in resultDict['trace'].varnames:
                available_vars.append(checkVar)
                available_varList.append(varList[ind])
            
        samples = pm.trace_to_dataframe(resultDict['trace'], varnames=available_vars)
    
        names, means, stds = [], [], []
        for oneVar in available_varList:
            mean=np.mean(samples[oneVar])
            std1 = np.std(samples[oneVar])
            #print("Var {},mean={}, std={}".format(oneVar,mean,std1))
            means.append(mean)
            stds.append(std1)
    
        t = Table()
        t['var name'] = available_varList
        t['mean'] = means
        t['std'] = stds
        if broadband == True:
            t.write('fit_results/broadband_fit_{}.csv'.format(self.descrip),overwrite=True)
        else:
            outDir = os.path.join('fit_results',self.descrip)
            if os.path.exists(outDir) == False:
                os.makedirs(outDir)
            t.write(os.path.join(outDir,'{}_wave_{}_fit.csv'.format(self.descrip,waveName)),overwrite=True)
        return t
    

    # def plot_trace(resultDict):
    #     _ = pm.traceplot(resultDict['trace'], var_names=["mean","logr_pl","b","u_star","rho_gp"])
    #     plt.savefig('plots/pymc3/traceplot.pdf')
    #     plt.close()

    def corner_plot(self,resultDict,compact=True,re_param_u=True,
                    truths=None,range=None):
        if re_param_u == True:
            limb_dark = "u_star_quadlimbdark__"
        else:
            limb_dark = 'u_star'
    
        if compact == True:
            varnames = ["depth"]
            outName = 'cornerplot_compact.pdf'
        else:
            varnames = ["depth","a","incl","t0"]
            outName = 'cornerplot_full.pdf'
        
        if self.u_lit == None:
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
        
        final_lc2D = flatten_chains(xdataset[lc_var_name])
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

    def run_all_broadband(self):
        modelDict = self.build_model()
        mapDict = self.find_mxap_with_clipping(modelDict)
        self.save_mxap_lc(mapDict)
        self.plot_test_point(mapDict)
        postDict = self.find_posterior(mapDict)
        self.print_es_summary(postDict)
        self.corner_plot(postDict)
        
        return postDict

    def run_all(self):
        self.run_all_broadband()
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
    
if __name__ == "__main__":
    freeze_support()
