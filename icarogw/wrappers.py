from .cupy_pal import get_module_array, get_module_array_scipy, np, sn
from .cosmology import alphalog_astropycosmology, cM_astropycosmology, extraD_astropycosmology, Xi0_astropycosmology, astropycosmology, eps0_astropycosmology
from .cosmology import  md_rate, md_gamma_rate, powerlaw_rate, beta_rate, beta_rate_line
from .priors import LowpassSmoothedProb, LowpassSmoothedProbEvolving, PowerLaw, BetaDistribution, TruncatedBetaDistribution, TruncatedGaussian, Bivariate2DGaussian, SmoothedPlusDipProb, BrokenPowerLawMultiPeak
from .priors import PowerLawGaussian, BrokenPowerLaw, PowerLawTwoGaussians, conditional_2dimpdf, conditional_2dimz_pdf, piecewise_constant_2d_distribution_normalized,paired_2dimpdf
from .priors import PowerLawStationary, PowerLawLinear, GaussianStationary, GaussianLinear, _mixed_linear_function, _mixed_double_sigmoid_function
from .priors import BrokenPowerLawTripleMultiPeak
from .priors import Bimodal
import copy
from astropy.cosmology import FlatLambdaCDM, FlatwCDM, Flatw0waCDM
import h5py

modgravity_wrappers = ['eps0_mod_wrap','Xi0_mod_wrap','extraD_mod_wrap',
                      'cM_mod_wrap','alphalog_mod_wrap']


lcdm_wrappers = ['FlatLambdaCDM_wrap','FlatwCDM_wrap',
                'Flatw0waCDM_wrap']

    
# A parent class for the rate
# LVK Reviewed
class rate_default(object):
    def evaluate(self,z):
        return self.rate.evaluate(z)
    def log_evaluate(self,z):
        return self.rate.log_evaluate(z)
    
# LVK Reviewed
class rateevolution_PowerLaw(rate_default):
    def __init__(self):
        self.population_parameters=['gamma']
    def update(self,**kwargs):
        self.rate=powerlaw_rate(**kwargs)

# LVK Reviewed
class rateevolution_Madau(rate_default):
    def __init__(self):
        self.population_parameters=['gamma','kappa','zp']
    def update(self,**kwargs):
        self.rate=md_rate(**kwargs)

class rateevolution_Madau_gamma(rate_default):
    def __init__(self):
        self.population_parameters=['gamma','kappa','zp','a','b','c']
    def update(self,**kwargs):
        self.rate=md_gamma_rate(**kwargs)

class rateevolution_beta(rate_default):
    def __init__(self):
        self.population_parameters=['a','b','c']
    def update(self,**kwargs):
        self.rate=beta_rate(**kwargs)

class rateevolution_beta_line(rate_default):
    def __init__(self):
        self.population_parameters=['a','b','c','d']
    def update(self,**kwargs):
        self.rate=beta_rate_line(**kwargs)


# LVK Reviewed
class FlatLambdaCDM_wrap(object):
    def __init__(self,zmax):
        self.population_parameters=['H0','Om0']
        self.cosmology=astropycosmology(zmax)
        self.astropycosmo=FlatLambdaCDM
    def update(self,**kwargs):
        self.cosmology.build_cosmology(self.astropycosmo(**kwargs))

# LVK Reviewed
class FlatwCDM_wrap(object):
    def __init__(self,zmax):
        self.population_parameters=['H0','Om0','w0']
        self.cosmology=astropycosmology(zmax)
        self.astropycosmo=FlatwCDM
    def update(self,**kwargs):
        self.cosmology.build_cosmology(self.astropycosmo(**kwargs))


class Flatw0waCDM_wrap(object):
    def __init__(self,zmax):
        self.population_parameters=['H0','Om0','w0','wa']
        self.cosmology=astropycosmology(zmax)
        self.astropycosmo=Flatw0waCDM
    def update(self,**kwargs):
        self.cosmology.build_cosmology(self.astropycosmo(**kwargs))

class eps0_mod_wrap(object):
    def __init__(self,bgwrap):
        self.bgwrap=copy.deepcopy(bgwrap)
        self.population_parameters=self.bgwrap.population_parameters+['eps0']
        self.cosmology=eps0_astropycosmology(bgwrap.cosmology.zmax)
    def update(self,**kwargs):
        bgdict={key:kwargs[key] for key in self.bgwrap.population_parameters}
        self.bgwrap.update(**bgdict)
        self.cosmology.build_cosmology(self.bgwrap.astropycosmo(**bgdict),eps0=kwargs['eps0'])

# LVK Reviewed
class Xi0_mod_wrap(object):
    def __init__(self,bgwrap):
        self.bgwrap=copy.deepcopy(bgwrap)
        self.population_parameters=self.bgwrap.population_parameters+['Xi0','n']
        self.cosmology=Xi0_astropycosmology(bgwrap.cosmology.zmax)
    def update(self,**kwargs):
        bgdict={key:kwargs[key] for key in self.bgwrap.population_parameters}
        self.bgwrap.update(**bgdict)
        self.cosmology.build_cosmology(self.bgwrap.astropycosmo(**bgdict),Xi0=kwargs['Xi0'],n=kwargs['n'])

# LVK Reviewed
class extraD_mod_wrap(object):
    def __init__(self,bgwrap):
        self.bgwrap=copy.deepcopy(bgwrap)
        self.population_parameters=self.bgwrap.population_parameters+['D','n','Rc']
        self.cosmology=extraD_astropycosmology(bgwrap.cosmology.zmax)
    def update(self,**kwargs):
        bgdict={key:kwargs[key] for key in self.bgwrap.population_parameters}
        self.bgwrap.update(**bgdict)
        self.cosmology.build_cosmology(self.bgwrap.astropycosmo(**bgdict),D=kwargs['D'],n=kwargs['n'],Rc=kwargs['Rc'])

# LVK Reviewed
class cM_mod_wrap(object):
    def __init__(self,bgwrap):
        self.bgwrap=copy.deepcopy(bgwrap)
        self.population_parameters=self.bgwrap.population_parameters+['cM']
        self.cosmology=cM_astropycosmology(bgwrap.cosmology.zmax)
    def update(self,**kwargs):
        bgdict={key:kwargs[key] for key in self.bgwrap.population_parameters}
        self.bgwrap.update(**bgdict)
        self.cosmology.build_cosmology(self.bgwrap.astropycosmo(**bgdict),cM=kwargs['cM'])

# LVK Reviewed
class alphalog_mod_wrap(object):
    def __init__(self,bgwrap):
        self.bgwrap=copy.deepcopy(bgwrap)
        self.population_parameters=self.bgwrap.population_parameters+['alphalog_1','alphalog_2','alphalog_3']
        self.cosmology=alphalog_astropycosmology(bgwrap.cosmology.zmax)
    def update(self,**kwargs):
        bgdict={key:kwargs[key] for key in self.bgwrap.population_parameters}
        self.bgwrap.update(**bgdict)
        self.cosmology.build_cosmology(self.bgwrap.astropycosmo(**bgdict),alphalog_1=kwargs['alphalog_1']
                                       ,alphalog_2=kwargs['alphalog_2'],alphalog_3=kwargs['alphalog_3'])

# A parent class for the standard 1D mass probabilities
class pm_prob(object):
    def pdf(self,mass_1_source):
        return self.prior.pdf(mass_1_source)
    def log_pdf(self,mass_1_source):
        return self.prior.log_pdf(mass_1_source)

class mass_ratio_prior_Gaussian(pm_prob):
    def __init__(self):
        self.population_parameters=['mu_q','sigma_q']
    def update(self,**kwargs):
        p1=TruncatedGaussian(kwargs['mu_q'],kwargs['sigma_q'],0.,1.)
        self.prior=p1

class mass_ratio_prior_Powerlaw(pm_prob):
    def __init__(self):
        self.population_parameters=['alpha_q']
    def update(self,**kwargs):
        self.prior=PowerLaw(0.,1.,kwargs['alpha_q'])

class lowSmoothedwrapper(pm_prob):
   def __init__(self, mw):
        self.population_parameters = ['delta_m'] + mw.population_parameters
        self.mw = mw
   def update(self,**kwargs):
        self.mw.update(**{key:kwargs[key] for key in self.mw.population_parameters})
        self.prior = LowpassSmoothedProb(self.mw.prior,kwargs['delta_m'])
 
# A parent class for the standard mass probabilities
# LVK Reviewed
class pm1m2_prob(object):
    def pdf(self,mass_1_source,mass_2_source):
        return self.prior.pdf(mass_1_source,mass_2_source)
    def log_pdf(self,mass_1_source,mass_2_source):
        return self.prior.log_pdf(mass_1_source,mass_2_source)
    
class pm1m2z_prob(object):
    def pdf(self,mass_1_source,mass_2_source,z):
        return self.prior.pdf(mass_1_source,mass_2_source,z)
    def log_pdf(self,mass_1_source,mass_2_source,z):
        return self.prior.log_pdf(mass_1_source,mass_2_source,z)

#LVK reviewed
class massprior_PowerLaw(pm_prob):
    def __init__(self):
        self.population_parameters=['alpha','mmin','mmax']
    def update(self,**kwargs):
        self.prior=PowerLaw(kwargs['mmin'],kwargs['mmax'],-kwargs['alpha'])
        
#LVK reviewed
class massprior_PowerLawPeak(pm_prob):
    def __init__(self):
        self.population_parameters=['alpha','mmin','mmax','mu_g','sigma_g','lambda_peak']
    def update(self,**kwargs):
        self.prior=PowerLawGaussian(kwargs['mmin'],kwargs['mmax'],-kwargs['alpha'],kwargs['lambda_peak'],kwargs['mu_g'],
                                         kwargs['sigma_g'],kwargs['mmin'],kwargs['mu_g']+5*kwargs['sigma_g'])
        
#LVK reviewed
class massprior_BrokenPowerLaw(pm_prob):
    def __init__(self):
        self.population_parameters=['alpha_1','alpha_2','mmin','mmax','b']
    def update(self,**kwargs):
        self.prior=BrokenPowerLaw(kwargs['mmin'],kwargs['mmax'],-kwargs['alpha_1'],-kwargs['alpha_2'],kwargs['b'])
        
#LVK reviewed
class massprior_MultiPeak(pm_prob):
    def __init__(self):
        self.population_parameters=['alpha','mmin','mmax','mu_g_low','sigma_g_low','lambda_g_low','mu_g_high','sigma_g_high','lambda_g']
    def update(self,**kwargs):
        self.prior=PowerLawTwoGaussians(kwargs['mmin'],kwargs['mmax'],-kwargs['alpha'],
                                             kwargs['lambda_g'],kwargs['lambda_g_low'],kwargs['mu_g_low'],
                                             kwargs['sigma_g_low'],kwargs['mmin'],kwargs['mu_g_low']+5*kwargs['sigma_g_low'],
                                             kwargs['mu_g_high'],kwargs['sigma_g_high'],kwargs['mmin'],kwargs['mu_g_high']+5*kwargs['sigma_g_high'])


#LVK reviewed
class massprior_BrokenPowerLawMultiPeak(pm_prob):
    def __init__(self):
        self.population_parameters=['alpha_1','alpha_2','mmin','mmax','b','mu_g_low','sigma_g_low','lambda_g_low','mu_g_high','sigma_g_high','lambda_g']
    def update(self,**kwargs):
        self.prior=BrokenPowerLawMultiPeak(kwargs['mmin'],kwargs['mmax'],-kwargs['alpha_1'],-kwargs['alpha_2'],kwargs['b'],
                                             kwargs['lambda_g'],kwargs['lambda_g_low'],kwargs['mu_g_low'],
                                             kwargs['sigma_g_low'],kwargs['mmin'],kwargs['mu_g_low']+5*kwargs['sigma_g_low'],
                                             kwargs['mu_g_high'],kwargs['sigma_g_high'],kwargs['mmin'],kwargs['mu_g_high']+5*kwargs['sigma_g_high'])

# New class for TripleMultipeak \(multipopulation model\)
class massprior_BrokenPowerLawTripleMultiPeak(pm_prob):
    def __init__(self):
        self.population_parameters=['alpha_1','alpha_2','mmin','mmax','b','mu_g_1','sigma_g_1','lambda_g','mu_g_2','sigma_g_2','lambda_1','mu_g_3','sigma_g_3','lambda_2']
    def update(self,**kwargs):
        self.prior=BrokenPowerLawTripleMultiPeak(kwargs['mmin'],kwargs['mmax'],-kwargs['alpha_1'],-kwargs['alpha_2'],kwargs['b'],
                                                 kwargs['lambda_g'],kwargs['lambda_1'],kwargs['lambda_2'],
                                                 kwargs['mu_g_1'],kwargs['sigma_g_1'],kwargs['mmin'],kwargs['mu_g_1']+5*kwargs['sigma_g_1'],
                                                 kwargs['mu_g_2'],kwargs['sigma_g_2'],kwargs['mmin'],kwargs['mu_g_2']+5*kwargs['sigma_g_2'],
                                                 kwargs['mu_g_3'],kwargs['sigma_g_3'],kwargs['mmin'],kwargs['mu_g_3']+5*kwargs['sigma_g_3'])


#LVK reviewed
class m1m2_conditioned(pm1m2_prob):
    def __init__(self,wrapper_m):
        self.population_parameters = wrapper_m.population_parameters+['beta']
        self.wrapper_m = wrapper_m
    def update(self,**kwargs):
        self.wrapper_m.update(**{key:kwargs[key] for key in self.wrapper_m.population_parameters})
        p1 = self.wrapper_m.prior
        p2 = PowerLaw(kwargs['mmin'],kwargs['mmax'],kwargs['beta'])
        self.prior=conditional_2dimpdf(p1,p2)

#LVK reviewed
class m1m2_conditioned_lowpass_m2(pm1m2z_prob):
    def __init__(self,wrapper_m):
        self.population_parameters = wrapper_m.population_parameters+['beta']
        self.wrapper_m = wrapper_m
    def update(self,**kwargs):
        self.wrapper_m.update(**{key:kwargs[key] for key in self.wrapper_m.population_parameters})
        p1 = self.wrapper_m.prior
        p2 = LowpassSmoothedProb(PowerLaw(kwargs['mmin'],kwargs['mmax'],kwargs['beta']),kwargs['delta_m'])
        self.prior=conditional_2dimz_pdf(p1,p2)

#LVK reviewed
class m1m2_conditioned_lowpass(pm1m2_prob):
    def __init__(self,wrapper_m):
        self.population_parameters = wrapper_m.population_parameters+['beta','delta_m']
        self.wrapper_m = wrapper_m
    def update(self,**kwargs):
        self.wrapper_m.update(**{key:kwargs[key] for key in self.wrapper_m.population_parameters})
        p1 = LowpassSmoothedProb(self.wrapper_m.prior,kwargs['delta_m'])
        p2 = LowpassSmoothedProb(PowerLaw(kwargs['mmin'],kwargs['mmax'],kwargs['beta']),kwargs['delta_m'])
        self.prior=conditional_2dimpdf(p1,p2)


#LVK reviewed
class m1m2_paired_massratio_dip(pm1m2_prob):
    def __init__(self,wrapper_m):
        self.population_parameters = wrapper_m.population_parameters + ['beta','bottomsmooth', 'topsmooth', 
                                                                        'leftdip','rightdip','leftdipsmooth', 
                                                                        'rightdipsmooth','deep']
        self.wrapper_m = wrapper_m
    def update(self,**kwargs):
        self.wrapper_m.update(**{key:kwargs[key] for key in self.wrapper_m.population_parameters})
        p = SmoothedPlusDipProb(self.wrapper_m.prior,**{key:kwargs[key] for key in ['bottomsmooth', 'topsmooth', 
                                                                        'leftdip', 'rightdip', 
                                                                        'leftdipsmooth','rightdipsmooth','deep']})
        def pairing_function(m1,m2,beta=kwargs['beta']):
            xp = get_module_array(m1)
            q = m2/m1
            toret = xp.power(q,beta)
            toret[q>1] = 0.
            return toret
        
        self.prior=paired_2dimpdf(p,pairing_function)


#LVK reviewed
class m1m2_paired_massratio_dip_general(pm1m2_prob):
    def __init__(self,wrapper_m):
        self.population_parameters = wrapper_m.population_parameters + ['beta_bottom','beta_top','bottomsmooth', 'topsmooth', 
                                                                        'leftdip','rightdip','leftdipsmooth', 
                                                                        'rightdipsmooth','deep']
        self.wrapper_m = wrapper_m
    def update(self,**kwargs):
        self.wrapper_m.update(**{key:kwargs[key] for key in self.wrapper_m.population_parameters})
        p = SmoothedPlusDipProb(self.wrapper_m.prior,**{key:kwargs[key] for key in ['bottomsmooth', 'topsmooth', 
                                                                        'leftdip', 'rightdip', 
                                                                        'leftdipsmooth','rightdipsmooth','deep']})
        
        def pairing_function(m1,m2,beta_bottom=kwargs['beta_bottom'],beta_top=kwargs['beta_top'],
                            rightdip=kwargs['rightdip']):
            
            xp = get_module_array(m1)
            q = m2/m1
            toret = xp.ones_like(q)
            idx = m2<=rightdip
            toret[idx] = xp.power(q[idx],beta_bottom)
            idx = m2>rightdip
            toret[idx] = xp.power(q[idx],beta_top)
            toret[q>1] = 0.
            return toret
        
        self.prior=paired_2dimpdf(p,pairing_function)


class m1m2_paired_massratio_bpl_dip_farah_2022(pm1m2_prob):
    def __init__(self):
        wrapper_m = massprior_BrokenPowerLaw()
        wrapper_m.population_parameters.remove('b')
        self.population_parameters = wrapper_m.population_parameters + ['beta_bottom','beta_top','bottomsmooth', 'topsmooth', 
                                                                        'leftdip','rightdip','leftdipsmooth', 
                                                                        'rightdipsmooth','deep']
        self.wrapper_m = wrapper_m
    def update(self,**kwargs):
        kwargs['b'] = (kwargs['leftdip']-kwargs['mmin'])/(kwargs['mmax']-kwargs['mmin'])
        self.wrapper_m.update(**{key:kwargs[key] for key in self.wrapper_m.population_parameters+['b']})
        p = SmoothedPlusDipProb(self.wrapper_m.prior,**{key:kwargs[key] for key in ['bottomsmooth', 'topsmooth', 
                                                                        'leftdip', 'rightdip', 
                                                                        'leftdipsmooth','rightdipsmooth','deep']})
        
        def pairing_function(m1,m2,beta_bottom=kwargs['beta_bottom'],beta_top=kwargs['beta_top']):
            
            xp = get_module_array(m1)
            q = m2/m1
            toret = xp.ones_like(q)
            idx = m2<=5.
            toret[idx] = xp.power(q[idx],beta_bottom)
            idx = m2>5.
            toret[idx] = xp.power(q[idx],beta_top)
            toret[q>1] = 0.
            return toret
        
        self.prior=paired_2dimpdf(p,pairing_function)

# Class for the BPLMultipopTriplePeak_dip
class m1m2_paired_bpl_triplepeak_dip(pm1m2_prob):
    def __init__(self):
        wrapper_m = massprior_BrokenPowerLawTripleMultiPeak()
        wrapper_m.population_parameters.remove('b')
        self.population_parameters = wrapper_m.population_parameters + ['beta_bottom','beta_top','bottomsmooth', 'topsmooth', 
                                                                        'leftdip','rightdip','leftdipsmooth', 
                                                                        'rightdipsmooth','deep']
        self.wrapper_m = wrapper_m
    
    def update(self,**kwargs):
        mbreak_NS = kwargs['leftdip'] + kwargs['leftdipsmooth']
        mbreak_BH = kwargs['rightdip'] - kwargs['rightdipsmooth']
        mbreak = 0.5*(mbreak_NS+mbreak_BH)
        kwargs['b'] = (mbreak-kwargs['mmin'])/(kwargs['mmax']-kwargs['mmin'])
        self.wrapper_m.update(**{key:kwargs[key] for key in self.wrapper_m.population_parameters+['b']})
        p = SmoothedPlusDipProb(self.wrapper_m.prior,**{key:kwargs[key] for key in ['bottomsmooth', 'topsmooth', 
                                                                        'leftdip', 'rightdip', 
                                                                        'leftdipsmooth','rightdipsmooth','deep']})
        
        def pairing_function(m1,m2,beta_bottom=kwargs['beta_bottom'],beta_top=kwargs['beta_top'],mbreak=mbreak):
            xp = get_module_array(m1)
            q = m2/m1
            toret = xp.ones_like(q)
            idx = m2<=mbreak
            toret[idx] = xp.power(q[idx],beta_bottom)
            idx = m2>mbreak
            toret[idx] = xp.power(q[idx],beta_top)
            toret[q>1] = 0.
            return toret
        
        self.prior=paired_2dimpdf(p,pairing_function)

#LVK reviewed
class m1m2_paired_massratio_bplmulti_dip(pm1m2_prob):
    def __init__(self):
        wrapper_m = massprior_BrokenPowerLawMultiPeak()
        wrapper_m.population_parameters.remove('b')
        self.population_parameters = wrapper_m.population_parameters + ['beta_bottom','beta_top','bottomsmooth', 'topsmooth', 
                                                                        'leftdip','rightdip','leftdipsmooth', 
                                                                        'rightdipsmooth','deep']
        self.wrapper_m = wrapper_m
    def update(self,**kwargs):
        mbreak_NS = kwargs['leftdip'] + kwargs['leftdipsmooth']
        mbreak_BH = kwargs['rightdip'] - kwargs['rightdipsmooth']
        mbreak = 0.5*(mbreak_NS+mbreak_BH)
        kwargs['b'] = (mbreak-kwargs['mmin'])/(kwargs['mmax']-kwargs['mmin'])
        self.wrapper_m.update(**{key:kwargs[key] for key in self.wrapper_m.population_parameters+['b']})
        p = SmoothedPlusDipProb(self.wrapper_m.prior,**{key:kwargs[key] for key in ['bottomsmooth', 'topsmooth', 
                                                                        'leftdip', 'rightdip', 
                                                                        'leftdipsmooth','rightdipsmooth','deep']})
        
        def pairing_function(m1,m2,beta_bottom=kwargs['beta_bottom'],beta_top=kwargs['beta_top'],mbreak=mbreak):
            # The motivation for using only m2 for beta top and bottom is that if m2 is a NS for sure 
            # it is more probable that the binary comes from isolated stellar binaries.
            xp = get_module_array(m1)
            q = m2/m1
            toret = xp.ones_like(q)
            idx = m2<=mbreak
            toret[idx] = xp.power(q[idx],beta_bottom)
            idx = m2>mbreak
            toret[idx] = xp.power(q[idx],beta_top)
            toret[q>1] = 0.
            return toret
        
        self.prior=paired_2dimpdf(p,pairing_function)



#LVK reviewed
class m1m2_paired_massratio_bplmulti_dip_conditioned(pm1m2_prob):
    def __init__(self):
        wrapper_m = massprior_BrokenPowerLawMultiPeak()
        wrapper_m.population_parameters.remove('b')
        self.population_parameters = wrapper_m.population_parameters + ['beta_bottom','beta_top','bottomsmooth', 'topsmooth', 
                                                                        'leftdip','rightdip','leftdipsmooth', 
                                                                        'rightdipsmooth','deep']
        self.wrapper_m = wrapper_m
    def update(self,**kwargs):
        mbreak_NS = kwargs['leftdip'] + kwargs['leftdipsmooth']
        mbreak_BH = kwargs['rightdip'] - kwargs['rightdipsmooth']
        mbreak = 0.5*(mbreak_NS+mbreak_BH)
        kwargs['b'] = (mbreak-kwargs['mmin'])/(kwargs['mmax']-kwargs['mmin'])
        self.wrapper_m.update(**{key:kwargs[key] for key in self.wrapper_m.population_parameters+['b']})
        p1 = SmoothedPlusDipProb(self.wrapper_m.prior,**{key:kwargs[key] for key in ['bottomsmooth', 'topsmooth', 
                                                                        'leftdip', 'rightdip', 
                                                                        'leftdipsmooth','rightdipsmooth','deep']})
        
        # Equivalent to a broken power law distribution in q = m2/m1
        bpl = BrokenPowerLaw(kwargs['mmin'],kwargs['mmax'],kwargs['beta_bottom'],kwargs['beta_top'],kwargs['b'])
        p2 = LowpassSmoothedProb(bpl,kwargs['bottomsmooth'])
        
        self.prior=conditional_2dimpdf(p1,p2)

#LVK reviewed
class m1m2_paired(pm1m2_prob):
    def __init__(self,wrapper_m):
        self.population_parameters = wrapper_m.population_parameters + ['beta']
        self.wrapper_m = wrapper_m
    def update(self,**kwargs):
        self.wrapper_m.update(**{key:kwargs[key] for key in self.wrapper_m.population_parameters})
    
        def pairing_function(m1,m2,beta=kwargs['beta']):
            xp = get_module_array(m1)
            q = m2/m1
            toret = xp.power(q,beta)
            toret[q>1] = 0.
            return toret
        self.prior=paired_2dimpdf(self.wrapper_m.prior,pairing_function)


class massprior_BinModel2d(pm1m2_prob):
    def __init__(self, n_bins_1d):
        self.population_parameters=['mmin','mmax']
        n_bins_total = int(n_bins_1d * (n_bins_1d + 1) / 2)
        self.bin_parameter_list = ['bin_' + str(i) for i in range(n_bins_total)]
        self.population_parameters += self.bin_parameter_list
    def update(self,**kwargs):
        kwargs_bin_parameters = np.array([kwargs[key] for key in self.bin_parameter_list])
        
        pdf_dist = piecewise_constant_2d_distribution_normalized(
            kwargs['mmin'], 
            kwargs['mmax'],
            kwargs_bin_parameters
        )
        
        self.prior=pdf_dist






# To be reviewed
class massprior_gaussian(pm_prob):
    def __init__(self):
        self.population_parameters=['mmin','mmax','mu_lambda','sigma_lambda_pop']

    def update(self,**kwargs):        
        self.prior=TruncatedGaussian(kwargs['mu_lambda'],kwargs['sigma_lambda_pop'],kwargs['mmin'],kwargs['mmax'])

# To be reviewed   
class m1m2_conditioned_gaussian(pm1m2_prob):
    def __init__(self,wrapper_m):
        self.population_parameters = wrapper_m.population_parameters
        self.wrapper_m = wrapper_m
    def update(self,**kwargs):
        self.wrapper_m.update(**{key:kwargs[key] for key in self.wrapper_m.population_parameters})
        p1 = self.wrapper_m.prior
        p2 = TruncatedGaussian(kwargs['mu_lambda'],kwargs['sigma_lambda_pop'],kwargs['mmin'],kwargs['mmax'])
        self.prior=conditional_2dimpdf(p1,p2)

# To be reviewed
class massprior_Bimodal(pm_prob):
    def __init__(self):
        self.population_parameters=['mmin','mmax','mu_low_lambda','sigma_low_lambda','mu_high_lambda','sigma_high_lambda','lambda_g_lambda']
    def update(self,**kwargs):
        self.prior=Bimodal(kwargs['mmin'],kwargs['mmax'],
                                             kwargs['lambda_g_lambda'],kwargs['mu_low_lambda'],
                                             kwargs['sigma_low_lambda'],kwargs['mmin'],kwargs['mmax'],
                                             kwargs['mu_high_lambda'],kwargs['sigma_high_lambda'],kwargs['mmin'],kwargs['mmax'])

# To be reviewed
class m1m2_conditioned_bimodal(pm1m2_prob):
    def __init__(self,wrapper_m):
        self.population_parameters = wrapper_m.population_parameters
        self.wrapper_m = wrapper_m
    def update(self,**kwargs):
        self.wrapper_m.update(**{key:kwargs[key] for key in self.wrapper_m.population_parameters})
        p1 = self.wrapper_m.prior
        p2 = Bimodal(kwargs['mmin'],kwargs['mmax'],
                                             kwargs['lambda_g_lambda'],kwargs['mu_low_lambda'],
                                             kwargs['sigma_low_lambda'],kwargs['mmin'],kwargs['mmax'],
                                             kwargs['mu_high_lambda'],kwargs['sigma_high_lambda'],kwargs['mmin'],kwargs['mmax'])
        self.prior=conditional_2dimpdf(p1,p2)



# ----------- #
# Spin models #
# ----------- #

class spinprior_default_evolving_gaussian(object):
    def __init__(self):
        self.population_parameters=['mu_chi','sigma_chi','mu_dot','sigma_dot'
                                    ,'sigma_t','csi_spin']
        self.event_parameters=['chi_1','chi_2','cos_t_1','cos_t_2']

    def update(self,**kwargs):
        self.mu_chi = kwargs['mu_chi']
        self.sigma_chi = kwargs['sigma_chi']
        self.mu_dot = kwargs['mu_dot']
        self.sigma_dot = kwargs['sigma_dot']     
        self.csi_spin = kwargs['csi_spin']
        self.aligned_pdf = TruncatedGaussian(1.,kwargs['sigma_t'],-1.,1.)

    def log_pdf(self,chi_1,chi_2,cos_t_1,cos_t_2,mass_1_source,mass_2_source):

        xp = get_module_array(chi_1)
        sx = get_module_array_scipy(chi_1)
 
        mu_chi_1 = self.mu_chi + self.mu_dot*mass_1_source
        sigma_chi_1 = self.sigma_chi + self.sigma_dot*mass_1_source
        mu_chi_2 = self.mu_chi + self.mu_dot*mass_2_source
        sigma_chi_2 = self.sigma_chi + self.sigma_dot*mass_2_source

        a, b = (0. - mu_chi_1) / sigma_chi_1, (1. - mu_chi_1) / sigma_chi_1 
        g1 = sx.stats.truncnorm.pdf(chi_1,a,b,loc=mu_chi_1,scale=sigma_chi_1)

        a, b = (0. - mu_chi_2) / sigma_chi_2, (1. - mu_chi_2) / sigma_chi_2 
        g2 = sx.stats.truncnorm.pdf(chi_2,a,b,loc=mu_chi_2,scale=sigma_chi_2)

        log_angular_part = xp.logaddexp(xp.log1p(-self.csi_spin)+xp.log(0.25),
                                    xp.log(self.csi_spin)+self.aligned_pdf.log_pdf(cos_t_1)+self.aligned_pdf.log_pdf(cos_t_2))

        out = xp.log(g1)+xp.log(g2)+log_angular_part
        
        return out
        
    def pdf(self,chi_1,chi_2,cos_t_1,cos_t_2,mass_1_source,mass_2_source):
        xp = get_module_array(chi_1)
        return xp.exp(self.log_pdf(chi_1,chi_2,cos_t_1,cos_t_2,mass_1_source,mass_2_source))

class spinprior_default_beta_window_gaussian(object):
    def __init__(self):
        self.population_parameters= ['mt', 
                                     'delta_mt','mix_f',
                                     'alpha_chi','beta_chi',
                                     'mu_chi','sigma_chi',
                                     'sigma_t','csi_spin']
        self.event_parameters=['chi_1','chi_2','cos_t_1','cos_t_2']
    

    def update(self,**kwargs):
        
        self.alpha_chi = kwargs['alpha_chi']
        self.beta_chi = kwargs['beta_chi']
        if (self.alpha_chi <= 1) | (self.beta_chi <= 1) :
            raise ValueError('Alpha and Beta must be > 1') 
        self.beta_pdf_chi = BetaDistribution(self.alpha_chi,self.beta_chi)
        
        self.mu_chi = kwargs['mu_chi']
        self.sigma_chi = kwargs['sigma_chi']
        self.csi_spin = kwargs['csi_spin']
        self.gaussian_pdf_chi = TruncatedGaussian(kwargs['mu_chi'],kwargs['sigma_chi'],0.,1.)

        self.mt, self.delta_mt, self.mix_f = kwargs['mt'], kwargs['delta_mt'], kwargs['mix_f']

        self.aligned_pdf = TruncatedGaussian(1.,kwargs['sigma_t'],-1.,1.)

    def log_pdf(self,chi_1,chi_2,cos_t_1,cos_t_2,mass_1_source,mass_2_source):
        
        xp = get_module_array(chi_1)
        # FIXME: The sigmoid function implementation has been changed. Check it is correct.
        wz_1 = _mixed_double_sigmoid_function(mass_1_source, self.mix_f, 0., self.mt, self.delta_mt)
        wz_2 = _mixed_double_sigmoid_function(mass_2_source, self.mix_f, 0., self.mt, self.delta_mt)

        pdf_1 = wz_1*self.beta_pdf_chi.pdf(chi_1)+(1-wz_1)*self.gaussian_pdf_chi.pdf(chi_1)
        pdf_2 = wz_2*self.beta_pdf_chi.pdf(chi_2)+(1-wz_2)*self.gaussian_pdf_chi.pdf(chi_2)

        log_angular_part = xp.logaddexp(xp.log1p(-self.csi_spin)+xp.log(0.25),
                                    xp.log(self.csi_spin)+self.aligned_pdf.log_pdf(cos_t_1)+self.aligned_pdf.log_pdf(cos_t_2))
        
        out = xp.log(pdf_1)+xp.log(pdf_2)+log_angular_part
        
        return out
        
    def pdf(self,chi_1,chi_2,cos_t_1,cos_t_2,mass_1_source,mass_2_source):
        xp = get_module_array(chi_1)
        return xp.exp(self.log_pdf(chi_1,chi_2,cos_t_1,cos_t_2,mass_1_source,mass_2_source))


class spinprior_default_beta_window_beta(object):
    def __init__(self):
        self.population_parameters= ['mt', 
                                     'delta_mt','mix_f',
                                     'alpha_chi_low','beta_chi_low',
                                     'alpha_chi_high','beta_chi_high',
                                     'sigma_t','csi_spin']
        self.event_parameters=['chi_1','chi_2','cos_t_1','cos_t_2']
    

    def update(self,**kwargs):
        
        self.alpha_chi_low = kwargs['alpha_chi_low']
        self.beta_chi_low = kwargs['beta_chi_low']
        self.alpha_chi_high = kwargs['alpha_chi_high']
        self.beta_chi_high = kwargs['beta_chi_high']
        self.csi_spin = kwargs['csi_spin']
        
        if (self.alpha_chi_low <= 1) | (self.beta_chi_low <= 1) | (self.alpha_chi_high <= 1) | (self.beta_chi_high <= 1):
            raise ValueError('Alpha and Beta must be > 1') 
        
        self.beta_pdf_chi_low = BetaDistribution(self.alpha_chi_low,self.beta_chi_low)
        self.beta_pdf_chi_high = BetaDistribution(self.alpha_chi_high,self.beta_chi_high)

        self.mt, self.delta_mt, self.mix_f = kwargs['mt'], kwargs['delta_mt'], kwargs['mix_f']

        self.aligned_pdf = TruncatedGaussian(1.,kwargs['sigma_t'],-1.,1.)

    def log_pdf(self,chi_1,chi_2,cos_t_1,cos_t_2,mass_1_source,mass_2_source):
        
        xp = get_module_array(chi_1)
        # FIXME: The sigmoid function implementation has been changed. Check it is correct.
        wz_1 = _mixed_double_sigmoid_function(mass_1_source, self.mix_f, 0., self.mt, self.delta_mt)
        wz_2 = _mixed_double_sigmoid_function(mass_2_source, self.mix_f, 0., self.mt, self.delta_mt)

        pdf_1 = wz_1*self.beta_pdf_chi_low.pdf(chi_1)+(1-wz_1)*self.beta_pdf_chi_high.pdf(chi_1)
        pdf_2 = wz_2*self.beta_pdf_chi_low.pdf(chi_2)+(1-wz_2)*self.beta_pdf_chi_high.pdf(chi_2)

        log_angular_part = xp.logaddexp(xp.log1p(-self.csi_spin)+xp.log(0.25),
                                    xp.log(self.csi_spin)+self.aligned_pdf.log_pdf(cos_t_1)+self.aligned_pdf.log_pdf(cos_t_2))
        
        out = xp.log(pdf_1)+xp.log(pdf_2)+log_angular_part
        
        return out
        
    def pdf(self,chi_1,chi_2,cos_t_1,cos_t_2,mass_1_source,mass_2_source):
        xp = get_module_array(chi_1)
        return xp.exp(self.log_pdf(chi_1,chi_2,cos_t_1,cos_t_2,mass_1_source,mass_2_source))

        
#LVK reviewed
class spinprior_default(object):
    def __init__(self):
        self.population_parameters=['alpha_chi','beta_chi','sigma_t','csi_spin']
        self.event_parameters=['chi_1','chi_2','cos_t_1','cos_t_2']
        self.name='DEFAULT'

    def update(self,**kwargs):
        self.alpha_chi = kwargs['alpha_chi']
        self.beta_chi = kwargs['beta_chi']
        self.csi_spin = kwargs['csi_spin']
        self.aligned_pdf = TruncatedGaussian(1.,kwargs['sigma_t'],-1.,1.)
        if (self.alpha_chi <= 1) | (self.beta_chi <= 1) :
            raise ValueError('Alpha and Beta must be > 1') 
        self.beta_pdf = BetaDistribution(self.alpha_chi,self.beta_chi)
    
    def log_pdf(self,chi_1,chi_2,cos_t_1,cos_t_2):
        xp = get_module_array(chi_1)
        log_angular_part = xp.logaddexp(xp.log1p(-self.csi_spin)+xp.log(0.25),
                                    xp.log(self.csi_spin)+self.aligned_pdf.log_pdf(cos_t_1)+self.aligned_pdf.log_pdf(cos_t_2))
        return self.beta_pdf.log_pdf(chi_1)+self.beta_pdf.log_pdf(chi_2)+log_angular_part
        
    def pdf(self,chi_1,chi_2,cos_t_1,cos_t_2):
        xp = get_module_array(chi_1)
        return xp.exp(self.log_pdf(chi_1,chi_2,cos_t_1,cos_t_2))


#LVK reviewed
class spinprior_default_gaussian(object):
    def __init__(self):
        self.population_parameters=['mu_chi_1','mu_chi_2','sigma_chi_1','sigma_chi_2','sigma_t','csi_spin']
        self.event_parameters=['chi_1','chi_2','cos_t_1','cos_t_2']

    def update(self,**kwargs):        
        self.csi_spin = kwargs['csi_spin']
        self.aligned_pdf = TruncatedGaussian(1.,kwargs['sigma_t'],-1.,1.)
        self.g1 = TruncatedGaussian(kwargs['mu_chi_1'],kwargs['sigma_chi_1'],0.,1.)
        self.g2 = TruncatedGaussian(kwargs['mu_chi_2'],kwargs['sigma_chi_2'],0.,1.)
    
    def log_pdf(self,chi_1,chi_2,cos_t_1,cos_t_2):
        xp = get_module_array(chi_1)
        log_angular_part = xp.logaddexp(xp.log1p(-self.csi_spin)+xp.log(0.25),
                                    xp.log(self.csi_spin)+self.aligned_pdf.log_pdf(cos_t_1)+self.aligned_pdf.log_pdf(cos_t_2))
        return self.g1.log_pdf(chi_1)+self.g2.log_pdf(chi_2)+log_angular_part
        
    def pdf(self,chi_1,chi_2,cos_t_1,cos_t_2):
        xp = get_module_array(chi_1)
        return xp.exp(self.log_pdf(chi_1,chi_2,cos_t_1,cos_t_2))



########### TGR Implementation ######################
class pseobprior_gaussian(object):
    def __init__(self):
        self.population_parameters=['mu_domega220','sigma_domega220','mu_dtau220','sigma_dtau220','rho_pseob']
    def update(self,**kwargs):
        # Note, the bounds below are set to be consistent with Lorenzo's injections
        self.pdf_evaluator=Bivariate2DGaussian(x1min=-10,x1max=10.,x1mean=kwargs['mu_domega220'],
                                               x2min=-10,x2max=10.,x2mean=kwargs['mu_dtau220'],
                                               x1variance=kwargs['sigma_domega220']**2.,x12covariance=kwargs['rho_pseob']*kwargs['sigma_domega220']*kwargs['sigma_dtau220'],
                                               x2variance=kwargs['sigma_dtau220']**2.)
    def log_pdf(self,domega220,dtau220):
        return self.pdf_evaluator.log_pdf(domega220,dtau220)
    def pdf(self,domega220,dtau220):
        xp = get_module_array(domega220)
        return xp.exp(self.log_pdf(domega220,dtau220))
#####################################################


# LVK Reviewed
class spinprior_gaussian(object):
    def __init__(self):
        self.population_parameters=['mu_chi_eff','sigma_chi_eff','mu_chi_p','sigma_chi_p','rho']
        self.event_parameters=['chi_eff','chi_p']
        self.name='GAUSSIAN'
    def update(self,**kwargs):
        self.pdf_evaluator=Bivariate2DGaussian(x1min=-1.,x1max=1.,x1mean=kwargs['mu_chi_eff'],
                                               x2min=0.,x2max=1.,x2mean=kwargs['mu_chi_p'],
                                               x1variance=kwargs['sigma_chi_eff']**2.,x12covariance=kwargs['rho']*kwargs['sigma_chi_eff']*kwargs['sigma_chi_p'],
                                               x2variance=kwargs['sigma_chi_p']**2.)
    def log_pdf(self,chi_eff,chi_p):
        return self.pdf_evaluator.log_pdf(chi_eff,chi_p)
    def pdf(self,chi_eff,chi_p):
        xp = get_module_array(chi_eff)
        return xp.exp(self.log_pdf(chi_eff,chi_p))
      
class spinprior_ECOs_totally_reflective(object):
    def __init__(self,q=1.):
        # q=1 is the polar case, q = 2 is the axial case, m=2 fixed
        self.q=q
        self.population_parameters=['alpha_chi','beta_chi','eps', 'f_eco', 'sigma_chi_ECO']
        self.event_parameters=['chi_1','chi_2'] 
        self.name='DEFAULT'
        
    def get_chi_crit(self, eps):
        xp = get_module_array(eps)   
        return xp.pi*(1.+self.q)/(2*xp.abs(xp.log(eps)))

    def update(self,**kwargs):
        self.alpha_chi = kwargs['alpha_chi']
        self.beta_chi = kwargs['beta_chi']
        self.eps = kwargs['eps']
        self.f_eco = kwargs['f_eco']
        self.sigma = kwargs['sigma_chi_ECO']
        self.chi_crit = self.get_chi_crit(self.eps)
        if (self.alpha_chi <= 1) | (self.beta_chi <= 1) :
            raise ValueError('Alpha and Beta must be > 1') 
            
        self.beta_pdf = BetaDistribution(self.alpha_chi,self.beta_chi)
        self.truncatedbeta_pdf = TruncatedBetaDistribution(self.alpha_chi,self.beta_chi,self.chi_crit)
        self.truncatedgaussian_pdf = TruncatedGaussian(self.chi_crit, self.sigma, 0., self.chi_crit)
        self.lambda_eco = 1-self.beta_pdf.cdf(np.array([self.get_chi_crit(self.eps)]))[0]
        
        
    def pdf(self,chi_1,chi_2):
        p_chi_1 = self.f_eco*((1-self.lambda_eco)*self.truncatedbeta_pdf.pdf(chi_1) + self.lambda_eco*self.truncatedgaussian_pdf.pdf(chi_1)) + (1-self.f_eco)*self.beta_pdf.pdf(chi_1) 
        p_chi_2 = self.f_eco*((1-self.lambda_eco)*self.truncatedbeta_pdf.pdf(chi_2) + self.lambda_eco*self.truncatedgaussian_pdf.pdf(chi_2)) + (1-self.f_eco)*self.beta_pdf.pdf(chi_2) 
        return p_chi_1*p_chi_2
        
        
    def log_pdf(self,chi_1,chi_2):
        xp = get_module_array(chi_1)
        return xp.log(self.pdf(chi_1,chi_2))
    

# ------------------------ #
# Redshift evolving models #
# ------------------------ #

class PowerLaw_PowerLaw():
    '''
        Class implementing the mass function model for two stationary PowerLaws.

        Some options are available:
            - flag_powerlaw_smoothing applies a left window function to the PowerLaws.

        The module is stand alone and not compatible with other wrappers.
    '''

    def __init__(self, flag_powerlaw_smoothing = 1):
        
        self.population_parameters   = ['alpha_a', 'mmin_a', 'mmax_a', 'alpha_b', 'mmin_b', 'mmax_b', 'mix']
        self.flag_powerlaw_smoothing = flag_powerlaw_smoothing

        if self.flag_powerlaw_smoothing: self.population_parameters += ['delta_m_a', 'delta_m_b']

    def update(self,**kwargs):

        self.alpha_a = kwargs['alpha_a']
        self.mmin_a  = kwargs['mmin_a']
        self.mmax_a  = kwargs['mmax_a']
        self.alpha_b = kwargs['alpha_b']
        self.mmin_b  = kwargs['mmin_b']
        self.mmax_b  = kwargs['mmax_b']
        self.mix     = kwargs['mix']

        if self.flag_powerlaw_smoothing:
            self.delta_m_a = kwargs['delta_m_a']
            self.delta_m_b = kwargs['delta_m_b']

    def pdf(self,m):

        xp = get_module_array(m)
        powerlaw_class_a = PowerLawStationary(self.alpha_a, self.mmin_a, self.mmax_a)
        powerlaw_class_b = PowerLawStationary(self.alpha_b, self.mmin_b, self.mmax_b)
        # Add left smoothing to the evolving PowerLaw.
        if self.flag_powerlaw_smoothing:
            powerlaw_class_a = LowpassSmoothedProb(powerlaw_class_a, self.delta_m_a)
            powerlaw_class_b = LowpassSmoothedProb(powerlaw_class_b, self.delta_m_b)
        powerlaw_part_a = powerlaw_class_a.pdf(m)
        powerlaw_part_b = powerlaw_class_b.pdf(m)

        # Impose the rate to be between [0,1].
        if (self.mix > 1) or (self.mix < 0):
            return xp.nan
        else:
            return self.mix * powerlaw_part_a + (1-self.mix) * powerlaw_part_b
    
    def log_pdf(self,m):
        xp = get_module_array(m)
        return xp.log(self.pdf(m))


class PowerLaw_PowerLaw_PowerLaw():
    '''
        Class implementing the mass function model for three stationary PowerLaws.

        Some options are available:
            - flag_powerlaw_smoothing applies a left window function to the PowerLaws.

        The module is stand alone and not compatible with other wrappers.
    '''

    def __init__(self, flag_powerlaw_smoothing = 1):
        
        self.population_parameters   = ['alpha_a', 'mmin_a', 'mmax_a', 'alpha_b', 'mmin_b', 'mmax_b', 'alpha_c', 'mmin_c', 'mmax_c', 'mix_alpha', 'mix_beta']
        self.flag_powerlaw_smoothing = flag_powerlaw_smoothing

        if self.flag_powerlaw_smoothing: self.population_parameters += ['delta_m_a', 'delta_m_b', 'delta_m_c']

    def update(self,**kwargs):

        self.alpha_a   = kwargs['alpha_a']
        self.mmin_a    = kwargs['mmin_a']
        self.mmax_a    = kwargs['mmax_a']
        self.alpha_b   = kwargs['alpha_b']
        self.mmin_b    = kwargs['mmin_b']
        self.mmax_b    = kwargs['mmax_b']
        self.alpha_c   = kwargs['alpha_c']
        self.mmin_c    = kwargs['mmin_c']
        self.mmax_c    = kwargs['mmax_c']
        self.mix_alpha = kwargs['mix_alpha']
        self.mix_beta  = kwargs['mix_beta']

        if self.flag_powerlaw_smoothing:
            self.delta_m_a = kwargs['delta_m_a']
            self.delta_m_b = kwargs['delta_m_b']
            self.delta_m_c = kwargs['delta_m_c']

    def pdf(self,m):

        xp = get_module_array(m)
        powerlaw_class_a = PowerLawStationary(self.alpha_a, self.mmin_a, self.mmax_a)
        powerlaw_class_b = PowerLawStationary(self.alpha_b, self.mmin_b, self.mmax_b)
        powerlaw_class_c = PowerLawStationary(self.alpha_c, self.mmin_c, self.mmax_c)
        # Add left smoothing to the evolving PowerLaw.
        if self.flag_powerlaw_smoothing:
            powerlaw_class_a = LowpassSmoothedProb(powerlaw_class_a, self.delta_m_a)
            powerlaw_class_b = LowpassSmoothedProb(powerlaw_class_b, self.delta_m_b)
            powerlaw_class_c = LowpassSmoothedProb(powerlaw_class_c, self.delta_m_c)
        powerlaw_part_a = powerlaw_class_a.pdf(m)
        powerlaw_part_b = powerlaw_class_b.pdf(m)
        powerlaw_part_c = powerlaw_class_c.pdf(m)

        # Impose the rate to be between [0,1].
        if (self.mix_alpha > 1) or (self.mix_alpha < 0) or (self.mix_beta > 1) or (self.mix_beta < 0) or (self.mix_alpha + self.mix_beta > 1):
            return xp.nan
        else:
            return self.mix_alpha * powerlaw_part_a + self.mix_beta * powerlaw_part_b + (1 - self.mix_beta - self.mix_alpha) * powerlaw_part_c

    def log_pdf(self,m):
        xp = get_module_array(m)
        return xp.log(self.pdf(m))


class PowerLaw_PowerLaw_Gaussian():
    '''
        Class implementing the mass function model for two stationary PowerLaws
        and a Gaussian peak.

        Some options are available:
            - flag_powerlaw_smoothing applies a left window function to the PowerLaws.

        The module is stand alone and not compatible with other wrappers.
    '''

    def __init__(self, flag_powerlaw_smoothing = 1):
        
        self.population_parameters   = ['alpha_a', 'mmin_a', 'mmax_a', 'alpha_b', 'mmin_b', 'mmax_b', 'mu_g', 'sigma_g', 'mix_alpha', 'mix_beta']
        self.flag_powerlaw_smoothing = flag_powerlaw_smoothing

        if self.flag_powerlaw_smoothing: self.population_parameters += ['delta_m_a', 'delta_m_b']

    def update(self,**kwargs):

        self.alpha_a   = kwargs['alpha_a']
        self.mmin_a    = kwargs['mmin_a']
        self.mmax_a    = kwargs['mmax_a']
        self.alpha_b   = kwargs['alpha_b']
        self.mmin_b    = kwargs['mmin_b']
        self.mmax_b    = kwargs['mmax_b']
        self.mu_g      = kwargs['mu_g']
        self.sigma_g   = kwargs['sigma_g']
        self.mix_alpha = kwargs['mix_alpha']
        self.mix_beta  = kwargs['mix_beta']

        if self.flag_powerlaw_smoothing:
            self.delta_m_a = kwargs['delta_m_a']
            self.delta_m_b = kwargs['delta_m_b']

    def pdf(self,m):

        xp = get_module_array(m)
        powerlaw_class_a = PowerLawStationary(self.alpha_a, self.mmin_a,  self.mmax_a)
        powerlaw_class_b = PowerLawStationary(self.alpha_b, self.mmin_b,  self.mmax_b)
        gaussian_class   = GaussianStationary(self.mu_g,    self.sigma_g, self.mmin_a)
        # Add left smoothing to the evolving PowerLaw.
        if self.flag_powerlaw_smoothing:
            powerlaw_class_a = LowpassSmoothedProb(powerlaw_class_a, self.delta_m_a)
            powerlaw_class_b = LowpassSmoothedProb(powerlaw_class_b, self.delta_m_b)
        powerlaw_part_a = powerlaw_class_a.pdf(m)
        powerlaw_part_b = powerlaw_class_b.pdf(m)
        gaussian_part   = gaussian_class.pdf(m)

        # Impose the rate to be between [0,1].
        if (self.mix_alpha > 1) or (self.mix_alpha < 0) or (self.mix_beta > 1) or (self.mix_beta < 0) or (self.mix_alpha + self.mix_beta > 1):
            return xp.nan
        else:
            return self.mix_alpha * powerlaw_part_a + self.mix_beta * powerlaw_part_b + (1 - self.mix_beta - self.mix_alpha) * gaussian_part
    
    def log_pdf(self,m):
        xp = get_module_array(m)
        return xp.log(self.pdf(m))


class PowerLaw_GaussianRedshiftLinear():
    '''
        Class implementing the mass function model conditioned on redshift p(m1|z),
        for a stationary PowerLaw and a redshift linearly-dependent Gaussian peak.

        Some options are available:
            - redshift_transition sets the function for the redshift transition
            between the PowerLaw and the Gaussian.
            - flag_powerlaw_smoothing applies a left window function to the PowerLaw.
            - flag_redshift_mixture allows for the transition function to evolve with redshift.

        The module is stand alone and not compatible with other wrappers.
    '''

    def __init__(self, redshift_transition = 'linear', flag_powerlaw_smoothing = 1, flag_redshift_mixture = 1):
        
        self.population_parameters   = ['alpha', 'mmin', 'mmax', 'mu_z0', 'mu_z1', 'sigma_z0', 'sigma_z1', 'mix_z0']
        self.redshift_transition     = redshift_transition
        self.flag_powerlaw_smoothing = flag_powerlaw_smoothing
        self.flag_redshift_mixture   = flag_redshift_mixture

        if self.flag_redshift_mixture:
            self.population_parameters += ['mix_z1']
            if self.redshift_transition == 'sigmoid': self.population_parameters += ['zt', 'delta_zt']
        if self.flag_powerlaw_smoothing: self.population_parameters += ['delta_m']

    def update(self,**kwargs):

        self.alpha    = kwargs['alpha']
        self.mmin     = kwargs['mmin']
        self.mmax     = kwargs['mmax']
        self.mu_z0    = kwargs['mu_z0']
        self.mu_z1    = kwargs['mu_z1']
        self.sigma_z0 = kwargs['sigma_z0']
        self.sigma_z1 = kwargs['sigma_z1']
        self.mix_z0   = kwargs['mix_z0']

        if self.flag_redshift_mixture:
            self.mix_z1 = kwargs['mix_z1']
            if self.redshift_transition == 'sigmoid': self.zt, self.delta_zt = kwargs['zt'], kwargs['delta_zt']
        if self.flag_powerlaw_smoothing: self.delta_m = kwargs['delta_m']

    def pdf(self,m,z):

        xp = get_module_array(m)
        if self.flag_redshift_mixture:
            if   self.redshift_transition == 'linear':
                wz = _mixed_linear_function(         z, self.mix_z0, self.mix_z1)
            elif self.redshift_transition == 'sigmoid':
                wz = _mixed_double_sigmoid_function( z, self.mix_z0, self.mix_z1, self.zt, self.delta_zt)
            else:
                raise ValueError('The slected redshift transition model {} does not exist. Exiting.'.format(self.redshift_transition))
        else:
            wz = self.mix_z0

        powerlaw_class = PowerLawStationary(self.alpha, self.mmin, self.mmax)
        # Add left smoothing to the evolving PowerLaw.
        if self.flag_powerlaw_smoothing: powerlaw_class = LowpassSmoothedProb(powerlaw_class, self.delta_m)
        gaussian_class = GaussianLinear(z, self.mu_z0, self.mu_z1, self.sigma_z0, self.sigma_z1, self.mmin)
        powerlaw_part  = powerlaw_class.pdf(m)
        gaussian_part  = gaussian_class.pdf(m)

        # Impose the rate to be between [0,1].
        if (xp.any(wz > 1)) or (xp.any(wz < 0)):
            return xp.nan
        else:
            return wz * powerlaw_part + (1-wz) * gaussian_part
    
    def log_pdf(self,m,z):
        xp = get_module_array(m)
        return xp.log(self.pdf(m,z))
    

class PowerLaw_GaussianRedshiftLinear_GaussianRedshiftLinear():
    '''
        Class implementing the mass function model conditioned on redshift p(m1|z),
        for a stationary PowerLaw and two redshift linearly-dependent Gaussian peaks.

        Some options are available:
            - redshift_transition sets the functions for the redshift transition
            between the PowerLaw and the two Gaussian peaks. The transition is
            the same between the PowerLaw and the first Gaussian a, and the two
            Gaussians a and b.
            - flag_powerlaw_smoothing applies a left window function to the PowerLaw.
            - flag_redshift_mixture allows for the transition functions to evolve with redshift.

        The module is stand alone and not compatible with other wrappers.
    '''

    def __init__(self, redshift_transition = 'linear', flag_powerlaw_smoothing = 1, flag_redshift_mixture = 1):
        
        self.population_parameters   = ['alpha', 'mmin', 'mmax', 'mu_a_z0', 'mu_a_z1', 'sigma_a_z0', 'sigma_a_z1', 'mu_b_z0', 'mu_b_z1', 'sigma_b_z0', 'sigma_b_z1', 'mix_alpha_z0', 'mix_beta_z0']
        self.redshift_transition     = redshift_transition
        self.flag_powerlaw_smoothing = flag_powerlaw_smoothing
        self.flag_redshift_mixture   = flag_redshift_mixture

        if self.flag_redshift_mixture:
            self.population_parameters += ['mix_alpha_z1', 'mix_beta_z1']
            if self.redshift_transition == 'sigmoid': self.population_parameters += ['zt', 'delta_zt']
        if self.flag_powerlaw_smoothing: self.population_parameters += ['delta_m']

    def update(self,**kwargs):

        self.alpha        = kwargs['alpha']
        self.mmin         = kwargs['mmin']
        self.mmax         = kwargs['mmax']
        self.mu_a_z0      = kwargs['mu_a_z0']
        self.mu_a_z1      = kwargs['mu_a_z1']
        self.sigma_a_z0   = kwargs['sigma_a_z0']
        self.sigma_a_z1   = kwargs['sigma_a_z1']
        self.mu_b_z0      = kwargs['mu_b_z0']
        self.mu_b_z1      = kwargs['mu_b_z1']
        self.sigma_b_z0   = kwargs['sigma_b_z0']
        self.sigma_b_z1   = kwargs['sigma_b_z1']
        self.mix_alpha_z0 = kwargs['mix_alpha_z0']
        self.mix_beta_z0  = kwargs['mix_beta_z0']

        if self.flag_redshift_mixture:
            self.mix_alpha_z1 = kwargs['mix_alpha_z1']
            self.mix_beta_z1  = kwargs['mix_beta_z1']
            if self.redshift_transition == 'sigmoid': self.zt, self.delta_zt = kwargs['zt'], kwargs['delta_zt']
        if self.flag_powerlaw_smoothing: self.delta_m = kwargs['delta_m']

    def pdf(self,m,z):

        xp = get_module_array(m)
        if self.flag_redshift_mixture:
            if   self.redshift_transition == 'linear':
                wz_alpha = _mixed_linear_function(         z, self.mix_alpha_z0, self.mix_alpha_z1)
                wz_beta  = _mixed_linear_function(         z, self.mix_beta_z0 , self.mix_beta_z1 )
            elif self.redshift_transition == 'sigmoid':
                wz_alpha = _mixed_double_sigmoid_function( z, self.mix_alpha_z0, self.mix_alpha_z1, self.zt, self.delta_zt)
                wz_beta  = _mixed_double_sigmoid_function( z, self.mix_beta_z0 , self.mix_beta_z1 , self.zt, self.delta_zt)
            else:
                raise ValueError('The slected redshift transition model {} does not exist. Exiting.'.format(self.redshift_transition))
        else:
            wz_alpha = self.mix_alpha_z0
            wz_beta  = self.mix_beta_z0

        powerlaw_class = PowerLawStationary(self.alpha, self.mmin, self.mmax)
        # Add left smoothing to the evolving PowerLaw.
        if self.flag_powerlaw_smoothing: powerlaw_class = LowpassSmoothedProb(powerlaw_class, self.delta_m)
        gaussian_a_class = GaussianLinear(z, self.mu_a_z0, self.mu_a_z1, self.sigma_a_z0, self.sigma_a_z1, self.mmin)
        gaussian_b_class = GaussianLinear(z, self.mu_b_z0, self.mu_b_z1, self.sigma_b_z0, self.sigma_b_z1, self.mmin)
        powerlaw_part    = powerlaw_class.pdf(m)
        gaussian_a_part  = gaussian_a_class.pdf(m)
        gaussian_b_part  = gaussian_b_class.pdf(m)

        # Impose the rate to be between [0,1].
        if (xp.any(wz_alpha > 1)) or (xp.any(wz_alpha < 0)) or (xp.any(wz_beta > 1)) or (xp.any(wz_beta < 0)) or (xp.any(wz_alpha + wz_beta > 1)):
            return xp.nan
        else:
            return wz_alpha * powerlaw_part + wz_beta * gaussian_a_part + (1 - wz_beta - wz_alpha) * gaussian_b_part
    
    def log_pdf(self,m,z):
        xp = get_module_array(m)
        return xp.log(self.pdf(m,z))


class PowerLawRedshiftLinear_GaussianRedshiftLinear():
    '''
        Class implementing the mass function model conditioned on redshift p(m1|z),
        for both a redshift linearly-dependent PowerLaw and Gaussian peak.

        Some options are available:
            - redshift_transition sets the function for the redshift transition
            between the PowerLaw and the Gaussian.
            - flag_powerlaw_smoothing applies a left window function to the PowerLaw.
            The smoothing slows heavily down the model evaluation.
            - flag_redshift_mixture allows for the transition function to evolve with redshift.

        The module is stand alone and not compatible with other wrappers.
    '''

    def __init__(self, redshift_transition = 'linear', flag_powerlaw_smoothing = 0, flag_redshift_mixture = 1):
        
        self.population_parameters   = ['alpha_z0', 'alpha_z1', 'mmin_z0', 'mmin_z1', 'mmax_z0', 'mmax_z1', 'mu_z0', 'mu_z1', 'sigma_z0', 'sigma_z1', 'mix_z0']
        self.redshift_transition     = redshift_transition
        self.flag_powerlaw_smoothing = flag_powerlaw_smoothing
        self.flag_redshift_mixture   = flag_redshift_mixture

        if self.flag_redshift_mixture:
            self.population_parameters += ['mix_z1']
            if self.redshift_transition == 'sigmoid': self.population_parameters += ['zt', 'delta_zt']
        if self.flag_powerlaw_smoothing: self.population_parameters += ['delta_m']

    def update(self,**kwargs):

        self.alpha_z0 = kwargs['alpha_z0']
        self.alpha_z1 = kwargs['alpha_z1']
        self.mmin_z0  = kwargs['mmin_z0']
        self.mmin_z1  = kwargs['mmin_z1']
        self.mmax_z0  = kwargs['mmax_z0']
        self.mmax_z1  = kwargs['mmax_z1']
        self.mu_z0    = kwargs['mu_z0']
        self.mu_z1    = kwargs['mu_z1']
        self.sigma_z0 = kwargs['sigma_z0']
        self.sigma_z1 = kwargs['sigma_z1']
        self.mix_z0   = kwargs['mix_z0']

        if self.flag_redshift_mixture:
            self.mix_z1 = kwargs['mix_z1']
            if self.redshift_transition == 'sigmoid': self.zt, self.delta_zt = kwargs['zt'], kwargs['delta_zt']
        if self.flag_powerlaw_smoothing: self.delta_m = kwargs['delta_m']

    def pdf(self,m,z):

        xp = get_module_array(m)
        if self.flag_redshift_mixture:
            if   self.redshift_transition == 'linear':
                wz = _mixed_linear_function(         z, self.mix_z0, self.mix_z1)
            elif self.redshift_transition == 'sigmoid':
                wz = _mixed_double_sigmoid_function( z, self.mix_z0, self.mix_z1, self.zt, self.delta_zt)
            else:
                raise ValueError('The slected redshift transition model {} does not exist. Exiting.'.format(self.redshift_transition))
        else:
            wz = self.mix_z0

        powerlaw_class = PowerLawLinear(z, self.alpha_z0, self.alpha_z1, self.mmin_z0, self.mmin_z1, self.mmax_z0, self.mmax_z1)
        # Add left smoothing to the evolving PowerLaw.
        # WARNING: The implementation is very slow, because the integral to normalise the windowed
        # distribution p(m1|z) needs to be computed at all redshifts corresponding to the PE samples and injections.
        if self.flag_powerlaw_smoothing: powerlaw_class = LowpassSmoothedProbEvolving(powerlaw_class, self.delta_m)
        gaussian_class = GaussianLinear(z, self.mu_z0, self.mu_z1, self.sigma_z0, self.sigma_z1, self.mmin_z0)
        powerlaw_part  = powerlaw_class.pdf(m)
        gaussian_part  = gaussian_class.pdf(m)

        # Impose the rate to be between [0,1].
        if (xp.any(wz > 1)) or (xp.any(wz < 0)):
            return xp.nan
        else:
            return wz * powerlaw_part + (1-wz) * gaussian_part
    
    def log_pdf(self,m,z):
        xp = get_module_array(m)
        return xp.log(self.pdf(m,z))
    

class PowerLawRedshiftLinear_PowerLawRedshiftLinear_PowerLawRedshiftLinear():
    '''
        Class implementing the mass function model conditioned on redshift p(m1|z),
        for three redshift linearly-dependent PowerLaws.

        Some options are available:
            - redshift_transition sets the function for the redshift transition
            between the PowerLaws (a, b and c).
            - flag_powerlaw_smoothing applies a left window function to the PowerLaws.
            The smoothing slows heavily down the model evaluation.
            - flag_redshift_mixture allows for the transition functions to evolve with redshift.

        The module is stand alone and not compatible with other wrappers.
    '''

    def __init__(self, redshift_transition = 'linear', flag_powerlaw_smoothing = 0, flag_redshift_mixture = 1):
        
        self.population_parameters   = ['alpha_a_z0', 'alpha_a_z1', 'mmin_a_z0', 'mmin_a_z1', 'mmax_a_z0', 'mmax_a_z1', 'alpha_b_z0', 'alpha_b_z1', 'mmin_b_z0', 'mmin_b_z1', 'mmax_b_z0', 'mmax_b_z1', 'alpha_c_z0', 'alpha_c_z1', 'mmin_c_z0', 'mmin_c_z1', 'mmax_c_z0', 'mmax_c_z1', 'mix_alpha_z0', 'mix_beta_z0']
        self.redshift_transition     = redshift_transition
        self.flag_powerlaw_smoothing = flag_powerlaw_smoothing
        self.flag_redshift_mixture   = flag_redshift_mixture

        if self.flag_redshift_mixture:
            self.population_parameters += ['mix_alpha_z1', 'mix_beta_z1']
        if self.flag_powerlaw_smoothing:
            self.population_parameters += ['delta_m_a', 'delta_m_b', 'delta_m_c']

    def update(self,**kwargs):

        self.alpha_a_z0   = kwargs['alpha_a_z0']
        self.alpha_a_z1   = kwargs['alpha_a_z1']
        self.mmin_a_z0    = kwargs['mmin_a_z0']
        self.mmin_a_z1    = kwargs['mmin_a_z1']
        self.mmax_a_z0    = kwargs['mmax_a_z0']
        self.mmax_a_z1    = kwargs['mmax_a_z1']
        self.alpha_b_z0   = kwargs['alpha_b_z0']
        self.alpha_b_z1   = kwargs['alpha_b_z1']
        self.mmin_b_z0    = kwargs['mmin_b_z0']
        self.mmin_b_z1    = kwargs['mmin_b_z1']
        self.mmax_b_z0    = kwargs['mmax_b_z0']
        self.mmax_b_z1    = kwargs['mmax_b_z1']
        self.alpha_c_z0   = kwargs['alpha_b_z0']
        self.alpha_c_z1   = kwargs['alpha_b_z1']
        self.mmin_c_z0    = kwargs['mmin_c_z0']
        self.mmin_c_z1    = kwargs['mmin_c_z1']
        self.mmax_c_z0    = kwargs['mmax_c_z0']
        self.mmax_c_z1    = kwargs['mmax_c_z1']
        self.mix_alpha_z0 = kwargs['mix_alpha_z0']
        self.mix_beta_z0  = kwargs['mix_beta_z0']

        if self.flag_redshift_mixture:
            self.mix_alpha_z1 = kwargs['mix_alpha_z1']
            self.mix_beta_z1  = kwargs['mix_beta_z1']
        if self.flag_powerlaw_smoothing:
            self.delta_m_a    = kwargs['delta_m_a']
            self.delta_m_b    = kwargs['delta_m_b']
            self.delta_m_c    = kwargs['delta_m_c']

    def pdf(self,m,z):

        xp = get_module_array(m)
        if self.flag_redshift_mixture:
            if   self.redshift_transition == 'linear':
                wz_alpha = _mixed_linear_function(z, self.mix_alpha_z0, self.mix_alpha_z1)
                wz_beta  = _mixed_linear_function(z, self.mix_beta_z0,  self.mix_beta_z1 )
            else:
                raise ValueError('The slected redshift transition model {} does not exist. Exiting.'.format(self.redshift_transition))
        else:
            wz_alpha = self.mix_alpha_z0
            wz_beta  = self.mix_beta_z0

        powerlaw_class_a = PowerLawLinear(z, self.alpha_a_z0, self.alpha_a_z1, self.mmin_a_z0, self.mmin_a_z1, self.mmax_a_z0, self.mmax_a_z1)
        powerlaw_class_b = PowerLawLinear(z, self.alpha_b_z0, self.alpha_b_z1, self.mmin_b_z0, self.mmin_b_z1, self.mmax_b_z0, self.mmax_b_z1)
        powerlaw_class_c = PowerLawLinear(z, self.alpha_c_z0, self.alpha_c_z1, self.mmin_c_z0, self.mmin_c_z1, self.mmax_c_z0, self.mmax_c_z1)
        # Add left smoothing to the evolving PowerLaw.
        # WARNING: The implementation is very slow, because the integral to normalise the windowed
        # distribution p(m1|z) needs to be computed at all redshifts corresponding to the PE samples and injections.
        if self.flag_powerlaw_smoothing:
            powerlaw_class_a = LowpassSmoothedProbEvolving(powerlaw_class_a, self.delta_m_a)
            powerlaw_class_b = LowpassSmoothedProbEvolving(powerlaw_class_b, self.delta_m_b)
            powerlaw_class_c = LowpassSmoothedProbEvolving(powerlaw_class_c, self.delta_m_c)
        powerlaw_part_a  = powerlaw_class_a.pdf(m)
        powerlaw_part_b  = powerlaw_class_b.pdf(m)
        powerlaw_part_c  = powerlaw_class_c.pdf(m)

        # Impose the rate to be between [0,1].
        if (xp.any(wz_alpha > 1)) or (xp.any(wz_alpha < 0)) or (xp.any(wz_beta > 1)) or (xp.any(wz_beta < 0)) or (xp.any(wz_alpha + wz_beta > 1)):
            return xp.nan
        else:
            return wz_alpha * powerlaw_part_a + wz_beta * powerlaw_part_b + (1 - wz_beta - wz_alpha) * powerlaw_part_c
    
    def log_pdf(self,m,z):
        xp = get_module_array(m)
        return xp.log(self.pdf(m,z))


class PowerLawRedshiftLinear_PowerLawRedshiftLinear_GaussianRedshiftLinear():
    '''
        Class implementing the mass function model conditioned on redshift p(m1|z),
        for both two redshift linearly-dependent PowerLaws and a Gaussian peak.

        Some options are available:
            - redshift_transition sets the function for the redshift transition
            between the PowerLaws (a and b) and the Gaussian.
            - flag_powerlaw_smoothing applies a left window function to the PowerLaws.
            The smoothing slows heavily down the model evaluation.
            - flag_redshift_mixture allows for the transition functions to evolve with redshift.

        The module is stand alone and not compatible with other wrappers.
    '''

    def __init__(self, redshift_transition = 'linear', flag_powerlaw_smoothing = 0, flag_redshift_mixture = 1):
        
        self.population_parameters   = ['alpha_a_z0', 'alpha_a_z1', 'mmin_a_z0', 'mmin_a_z1', 'mmax_a_z0', 'mmax_a_z1', 'alpha_b_z0', 'alpha_b_z1', 'mmin_b_z0', 'mmin_b_z1', 'mmax_b_z0', 'mmax_b_z1', 'mu_z0', 'mu_z1', 'sigma_z0', 'sigma_z1', 'mix_alpha_z0', 'mix_beta_z0']
        self.redshift_transition     = redshift_transition
        self.flag_powerlaw_smoothing = flag_powerlaw_smoothing
        self.flag_redshift_mixture   = flag_redshift_mixture

        if self.flag_redshift_mixture:
            self.population_parameters += ['mix_alpha_z1', 'mix_beta_z1']
        if self.flag_powerlaw_smoothing:
            self.population_parameters += ['delta_m_a', 'delta_m_b']

    def update(self,**kwargs):

        self.alpha_a_z0   = kwargs['alpha_a_z0']
        self.alpha_a_z1   = kwargs['alpha_a_z1']
        self.mmin_a_z0    = kwargs['mmin_a_z0']
        self.mmin_a_z1    = kwargs['mmin_a_z1']
        self.mmax_a_z0    = kwargs['mmax_a_z0']
        self.mmax_a_z1    = kwargs['mmax_a_z1']
        self.alpha_b_z0   = kwargs['alpha_b_z0']
        self.alpha_b_z1   = kwargs['alpha_b_z1']
        self.mmin_b_z0    = kwargs['mmin_b_z0']
        self.mmin_b_z1    = kwargs['mmin_b_z1']
        self.mmax_b_z0    = kwargs['mmax_b_z0']
        self.mmax_b_z1    = kwargs['mmax_b_z1']
        self.mu_z0        = kwargs['mu_z0']
        self.mu_z1        = kwargs['mu_z1']
        self.sigma_z0     = kwargs['sigma_z0']
        self.sigma_z1     = kwargs['sigma_z1']
        self.mix_alpha_z0 = kwargs['mix_alpha_z0']
        self.mix_beta_z0  = kwargs['mix_beta_z0']

        if self.flag_redshift_mixture:
            self.mix_alpha_z1 = kwargs['mix_alpha_z1']
            self.mix_beta_z1  = kwargs['mix_beta_z1']
        if self.flag_powerlaw_smoothing:
            self.delta_m_a    = kwargs['delta_m_a']
            self.delta_m_b    = kwargs['delta_m_b']

    def pdf(self,m,z):

        xp = get_module_array(m)
        if self.flag_redshift_mixture:
            if   self.redshift_transition == 'linear':
                wz_alpha = _mixed_linear_function(z, self.mix_alpha_z0, self.mix_alpha_z1)
                wz_beta  = _mixed_linear_function(z, self.mix_beta_z0,  self.mix_beta_z1 )
            else:
                raise ValueError('The slected redshift transition model {} does not exist. Exiting.'.format(self.redshift_transition))
        else:
            wz_alpha = self.mix_alpha_z0
            wz_beta  = self.mix_beta_z0

        powerlaw_class_a = PowerLawLinear(z, self.alpha_a_z0, self.alpha_a_z1, self.mmin_a_z0, self.mmin_a_z1, self.mmax_a_z0, self.mmax_a_z1)
        powerlaw_class_b = PowerLawLinear(z, self.alpha_b_z0, self.alpha_b_z1, self.mmin_b_z0, self.mmin_b_z1, self.mmax_b_z0, self.mmax_b_z1)
        gaussian_class   = GaussianLinear(z, self.mu_z0, self.mu_z1, self.sigma_z0, self.sigma_z1, self.mmin_a_z0)
        # Add left smoothing to the evolving PowerLaw.
        # WARNING: The implementation is very slow, because the integral to normalise the windowed
        # distribution p(m1|z) needs to be computed at all redshifts corresponding to the PE samples and injections.
        if self.flag_powerlaw_smoothing:
            powerlaw_class_a = LowpassSmoothedProbEvolving(powerlaw_class_a, self.delta_m_a)
            powerlaw_class_b = LowpassSmoothedProbEvolving(powerlaw_class_b, self.delta_m_b)
        powerlaw_part_a  = powerlaw_class_a.pdf(m)
        powerlaw_part_b  = powerlaw_class_b.pdf(m)
        gaussian_part    = gaussian_class.pdf(m)
    
        # Impose the rate to be between [0,1].
        if (xp.any(wz_alpha > 1)) or (xp.any(wz_alpha < 0)) or (xp.any(wz_beta > 1)) or (xp.any(wz_beta < 0)) or (xp.any(wz_alpha + wz_beta > 1)):
            return xp.nan
        else:
            return wz_alpha * powerlaw_part_a + wz_beta * powerlaw_part_b + (1 - wz_beta - wz_alpha) * gaussian_part
    
    def log_pdf(self,m,z):
        xp = get_module_array(m)
        return xp.log(self.pdf(m,z))


class GaussianRedshiftLinear_GaussianRedshiftLinear():
    '''
        Class implementing the mass function model conditioned on redshift p(m1|z),
        for two redshift linearly-dependent Gaussian peaks.

        Some options are available:
            - redshift_transition sets the function for the redshift transition
            between the PowerLaw and the Gaussian.
            - flag_redshift_mixture allows for the transition function to evolve with redshift.

        The module is stand alone and not compatible with other wrappers.
    '''

    def __init__(self, redshift_transition = 'linear', flag_redshift_mixture = 1):
        
        self.population_parameters = ['mu_a_z0', 'mu_a_z1', 'sigma_a_z0', 'sigma_a_z1', 'mu_b_z0', 'mu_b_z1', 'sigma_b_z0', 'sigma_b_z1', 'mix_z0', 'mmin_g']
        self.redshift_transition   = redshift_transition
        self.flag_redshift_mixture = flag_redshift_mixture
        
        if self.flag_redshift_mixture:
            self.population_parameters += ['mix_z1']
            if self.redshift_transition == 'sigmoid': self.population_parameters += ['zt', 'delta_zt']

    def update(self,**kwargs):

        self.mu_a_z0    = kwargs['mu_a_z0']
        self.mu_a_z1    = kwargs['mu_a_z1']
        self.sigma_a_z0 = kwargs['sigma_a_z0']
        self.sigma_a_z1 = kwargs['sigma_a_z1']
        self.mu_b_z0    = kwargs['mu_b_z0']
        self.mu_b_z1    = kwargs['mu_b_z1']
        self.sigma_b_z0 = kwargs['sigma_b_z0']
        self.sigma_b_z1 = kwargs['sigma_b_z1']
        self.mix_z0     = kwargs['mix_z0']
        self.mmin_g     = kwargs['mmin_g']

        if self.flag_redshift_mixture:
            self.mix_z1 = kwargs['mix_z1']
            if self.redshift_transition == 'sigmoid': self.zt, self.delta_zt = kwargs['zt'], kwargs['delta_zt']

    def pdf(self,m,z):

        xp = get_module_array(m)
        if self.flag_redshift_mixture:
            if   self.redshift_transition == 'linear':
                wz = _mixed_linear_function(         z, self.mix_z0, self.mix_z1)
            elif self.redshift_transition == 'sigmoid':
                wz = _mixed_double_sigmoid_function( z, self.mix_z0, self.mix_z1, self.zt, self.delta_zt)
            else:
                raise ValueError('The slected redshift transition model {} does not exist. Exiting.'.format(self.redshift_transition))
        else:
            wz = self.mix_z0

        gaussian_a_class = GaussianLinear(z, self.mu_a_z0, self.mu_a_z1, self.sigma_a_z0, self.sigma_a_z1, self.mmin_g)
        gaussian_b_class = GaussianLinear(z, self.mu_b_z0, self.mu_b_z1, self.sigma_b_z0, self.sigma_b_z1, self.mmin_g)
        gaussian_a_part  = gaussian_a_class.pdf(m)
        gaussian_b_part  = gaussian_b_class.pdf(m)

        # Impose the rate to be between [0,1].
        if (xp.any(wz > 1)) or (xp.any(wz < 0)):
            return xp.nan
        else:
            return wz * gaussian_a_part + (1-wz) * gaussian_b_part
    
    def log_pdf(self,m,z):
        xp = get_module_array(m)
        return xp.log(self.pdf(m,z))


class GaussianRedshiftLinear_GaussianRedshiftLinear_GaussianRedshiftLinear():
    '''
        Class implementing the mass function model conditioned on redshift p(m1|z),
        for three redshift linearly-dependent Gaussian peaks.

        Some options are available:
            - redshift_transition sets the function for the redshift transition
            between the three Gaussian peaks. The transition is the same between the
            first two Gaussians a and b, and the other two Gaussians b and c.
            - flag_redshift_mixture allows for the transition functions to evolve with redshift.

        The module is stand alone and not compatible with other wrappers.
    '''

    def __init__(self, redshift_transition = 'linear', flag_redshift_mixture = 1):

        self.population_parameters = ['mu_a_z0', 'mu_a_z1', 'sigma_a_z0', 'sigma_a_z1', 'mu_b_z0', 'mu_b_z1', 'sigma_b_z0', 'sigma_b_z1', 'mu_c_z0', 'mu_c_z1', 'sigma_c_z0', 'sigma_c_z1', 'mix_alpha_z0', 'mix_beta_z0', 'mmin_g']
        self.redshift_transition   = redshift_transition
        self.flag_redshift_mixture = flag_redshift_mixture

        if self.flag_redshift_mixture:
            self.population_parameters += ['mix_alpha_z1', 'mix_beta_z1']
            if self.redshift_transition == 'sigmoid': self.population_parameters += ['zt', 'delta_zt']

    def update(self,**kwargs):

        self.mu_a_z0      = kwargs['mu_a_z0']
        self.mu_a_z1      = kwargs['mu_a_z1']
        self.sigma_a_z0   = kwargs['sigma_a_z0']
        self.sigma_a_z1   = kwargs['sigma_a_z1']
        self.mu_b_z0      = kwargs['mu_b_z0']
        self.mu_b_z1      = kwargs['mu_b_z1']
        self.sigma_b_z0   = kwargs['sigma_b_z0']
        self.sigma_b_z1   = kwargs['sigma_b_z1']
        self.mu_c_z0      = kwargs['mu_c_z0']
        self.mu_c_z1      = kwargs['mu_c_z1']
        self.sigma_c_z0   = kwargs['sigma_c_z0']
        self.sigma_c_z1   = kwargs['sigma_c_z1']
        self.mix_alpha_z0 = kwargs['mix_alpha_z0']
        self.mix_beta_z0  = kwargs['mix_beta_z0']
        self.mmin_g       = kwargs['mmin_g']

        if self.flag_redshift_mixture:
            self.mix_alpha_z1 = kwargs['mix_alpha_z1']
            self.mix_beta_z1  = kwargs['mix_beta_z1']
            if self.redshift_transition == 'sigmoid': self.zt, self.delta_zt = kwargs['zt'], kwargs['delta_zt']

    def pdf(self,m,z):

        xp = get_module_array(m)
        if self.flag_redshift_mixture:
            if   self.redshift_transition == 'linear':
                wz_alpha = _mixed_linear_function(         z, self.mix_alpha_z0, self.mix_alpha_z1)
                wz_beta  = _mixed_linear_function(         z, self.mix_beta_z0 , self.mix_beta_z1 )
            elif self.redshift_transition == 'sigmoid':
                wz_alpha = _mixed_double_sigmoid_function( z, self.mix_alpha_z0, self.mix_alpha_z1, self.zt, self.delta_zt)
                wz_beta  = _mixed_double_sigmoid_function( z, self.mix_beta_z0 , self.mix_beta_z1 , self.zt, self.delta_zt)
            else:
                raise ValueError('The slected redshift transition model {} does not exist. Exiting.'.format(self.redshift_transition))
        else:
            wz_alpha = self.mix_alpha_z0
            wz_beta  = self.mix_beta_z0

        gaussian_a_class = GaussianLinear(z, self.mu_a_z0, self.mu_a_z1, self.sigma_a_z0, self.sigma_a_z1, self.mmin_g)
        gaussian_b_class = GaussianLinear(z, self.mu_b_z0, self.mu_b_z1, self.sigma_b_z0, self.sigma_b_z1, self.mmin_g)
        gaussian_c_class = GaussianLinear(z, self.mu_c_z0, self.mu_c_z1, self.sigma_c_z0, self.sigma_c_z1, self.mmin_g)
        gaussian_a_part  = gaussian_a_class.pdf(m)
        gaussian_b_part  = gaussian_b_class.pdf(m)
        gaussian_c_part  = gaussian_c_class.pdf(m)

        # Impose the rate to be between [0,1].
        if   (xp.any(wz_alpha > 1)) or (xp.any(wz_alpha < 0)) or (xp.any(wz_beta > 1)) or (xp.any(wz_beta < 0)) or (xp.any(wz_alpha + wz_beta > 1)):
            return xp.nan
        else:
            return wz_alpha * gaussian_a_part + wz_beta * gaussian_b_part + (1 - wz_beta - wz_alpha) * gaussian_c_part

    def log_pdf(self,m,z):
        xp = get_module_array(m)
        return xp.log(self.pdf(m,z))


class GaussianEvolving():
    '''
        Class implementing the mass function model conditioned on redshift p(m|z), for
        one redshift evolving Gaussian peak with arbitrary polynomial redshift expansion.

        Some options are available:
            - order sets the order of the redshift expansion. The population
            parameters [mu_zx, sigma_zx] are automatically defined from the
            selected order.
            Ex. order = 2 gives mu_z = mu_z0 + mu_z1*z + mu_z2*z^2

        The module is stand alone and not compatible with other wrappers.
    '''

    def __init__(self, order = 1):

        self.order = order
        mu_list    = ['mu_z{}'.format(   i) for i in range(order + 1)]
        sigma_list = ['sigma_z{}'.format(i) for i in range(order + 1)]
        self.population_parameters = mu_list + sigma_list

    def update(self, **kwargs):

        for par in kwargs.keys():
            globals()['self.%s' % par] = kwargs[par]

    def polynomial(self, expansion_order, x, variable):
        
        pol = 0
        for i in range(expansion_order + 1):
            par = '{}_z{}'.format(variable, i)
            pol += globals()['self.%s' % par] * x ** i
        return pol

    def log_pdf(self, m, z):

        xp = get_module_array(m)
        sx = get_module_array_scipy(m)
        self.muz    = self.polynomial(self.order, z, 'mu')
        self.sigmaz = self.polynomial(self.order, z, 'sigma')
        a, b = (0. - self.muz) / self.sigmaz, (self.muz + 6*self.muz - self.muz) / self.sigmaz  # Truncte the Gaussian at zero and mu+6*sigma. This improve the numerical stability.
        gaussian = xp.log( sx.stats.truncnorm.pdf(m, a, b, loc = self.muz, scale = self.sigmaz) )
        return gaussian

    def pdf(self, m, z):
        xp = get_module_array(m)
        return xp.exp(self.log_pdf(m, z))
    
    def return_mu_sigma(self):
        return self.muz, self.sigmaz
    


# ------------------------ #
# Equation of State models #
# ------------------------ #

# To be reviewed
class lambdaprior_default(object):
    '''
    Class implementing the equation of state function model linking the source masses to the tidal deformabilities of neutron stars.

    This module considered a non-paramettric EOS: either given by a HDF5 dataset with 'M' and 'Lambda' parameters 
    or a csv file with the same columns.     

    Notes that the update_table function must be used before giving this wrapper to a CBC rate model. 
    '''
    def __init__(self):
        # Parameter list for EOS:
        self.population_parameters=['sigma_lambda']

        self.table = None
        
        self.event_parameters=['lambda_1','lambda_2']
        self.name='DEFAULT EOS'
    

    def update(self,**kwargs):
        self.sigma_lambda = kwargs['sigma_lambda']


    def update_table(self, filename):

        if isinstance(filename, str):

            if filename.endswith(".csv"):
                import pandas
    
                data_eos = pandas.read_csv(filename, usecols=['M', 'Lambda'])
                self.table = data_eos

        elif isinstance(filename, h5py.Dataset):
            self.table = filename

        else:
            raise TypeError('You must give a csv file or a hdf5 dataset such that "M" and "Lambda" gives respectively the neutron star mass and its tidal deformability.')
        

        # Keep only the stable branch
        masses = self.table['M']
        lambdas = self.table['Lambda']

        imax = np.argmax(masses)
        m1 = masses[:imax + 1]
        l1 = lambdas[:imax + 1]

        imin = np.argmin(m1)
        m2 = m1[imin:]
        l2 = l1[imin:]
        
        self.table['M'] = m2
        self.table['Lambda'] = l2


   
    def EOS_L2ms(self,Lambda):
        xp = get_module_array(Lambda)

        idx = xp.argsort(self.table['Lambda'])
        data_lambda = self.table['Lambda']
        data_M = self.table['M']
        msource = xp.interp(Lambda, data_lambda[idx], data_M[idx])

        return msource
    
    def EOS_ms2L(self,msource):
        xp = get_module_array(msource)

        idx = xp.argsort(self.table['M'])
        data_lambda = self.table['Lambda']
        data_M = self.table['M']
        Lambda = xp.interp(msource, data_M[idx], data_lambda[idx])

        return Lambda
        

    
    def log_pdf(self,lambda_1,lambda_2, msource1, msource2):
        xp = get_module_array(lambda_1)
        
        Lambda_from_mass_1 = self.EOS_ms2L(msource1)
        Lambda_from_mass_2 = self.EOS_ms2L(msource2)

        gauss_1 = -(xp.log(Lambda_from_mass_1)-xp.log(lambda_1))**2/(2*self.sigma_lambda**2) - xp.log(xp.sqrt(2*xp.pi)*self.sigma_lambda) - xp.log(lambda_1)
        gauss_2 = -(xp.log(Lambda_from_mass_2)-xp.log(lambda_2))**2/(2*self.sigma_lambda**2) - xp.log(xp.sqrt(2*xp.pi)*self.sigma_lambda) - xp.log(lambda_2)
   
        return gauss_1 + gauss_2 
        

    def pdf(self,lambda_1,lambda_2, msource1, msource2):
        xp = get_module_array(lambda_1)
        return xp.exp(self.log_pdf(lambda_1,lambda_2, msource1, msource2))
    



# To be reviewed
class lambdaprior_parametric(object):
    '''
    Class implementing the equation of state function model linking the source masses to the tidal deformabilities of neutron stars.

    This module considered a paramettric EOS: it will solve the TOV equations.
    It considers: a polytropic P(rho) for the crust (Gamma_crust), the nuclear expression ('rho_0', 'e0_rho0', 'K0', 'e_sym_0', 'L', 'Ksym')
    for the core and three polytropics at higher densities ('Gamma_1', 'Gamma_2', 'Gamma_3'). 
    The density transitions are given by 'n_min' (minimum density), 'n_t' (crust-core), 'n_t_1' (core-polytropic one), 
    'n_t_2' (polytropic one - polytropic two), 'n_t_3' (polytropic two - polytropic three), 'n_max' (maximum density).
    '''
    def __init__(self):
        # Parameter list for EOS:
        self.population_parameters=['sigma_lambda', 'rho_0', 'e0_rho0', 'K0', 'e_sym_0', 'L', 'Ksym', 'Gamma_crust', 'Gamma_1', 'Gamma_2', 'Gamma_3', 'n_min', 'n_t', 'n_t_1', 'n_t_2', 'n_t_3', 'n_max']

        self.Lambda_table = None
        self.mass_table = None
        
        self.event_parameters=['lambda_1','lambda_2']
        self.name='PARAMETRIC EOS'
    

    def update(self,**kwargs):
        # Parameters for EOS:
        self.sigma_lambda = kwargs['sigma_lambda']

        self.rho_0 = kwargs['rho_0']
        self.e0_rho0 = kwargs['e0_rho0']
        self.K0 = kwargs['K0']
        self.e_sym_0 = kwargs['e_sym_0']
        self.L = kwargs['L']
        self.Ksym = kwargs['Ksym']
        self.Gamma_crust = kwargs['Gamma_crust']
        self.Gamma_1 = kwargs['Gamma_1']
        self.Gamma_2 = kwargs['Gamma_2']
        self.Gamma_3 = kwargs['Gamma_3']
        self.n_min = kwargs['n_min']
        self.n_t = kwargs['n_t']
        self.n_t_1 = kwargs['n_t_1']
        self.n_t_2 = kwargs['n_t_2']
        self.n_t_3 = kwargs['n_t_3']
        self.n_max = kwargs['n_max']

        # Construction of the EOS 
        try:
            nn, PP, EE = self.EOS_grid()

        except (ValueError):
            raise TypeError('Fail to construct the EOS. Please verifiy the EOS parameters.')


        # When crust part is not necessary because nuclear part goes to the surface (pressure = 0)
        G_SI = 6.67430e-11
        C_SI = 299_792_458.0
        MEVFM3_TO_GEOM = 1.602176634e32 * G_SI / C_SI**4 * 1.0e6
        idx_nt = np.where(nn == self.n_t)[0]
        if np.any(PP[idx_nt[0]:]/MEVFM3_TO_GEOM < 1e-11):
            idx_pp = np.where(PP/MEVFM3_TO_GEOM < 1e-11)[0]
            PP = PP[idx_pp[-1]:]
            EE = EE[idx_pp[-1]:]
            nn = nn[idx_pp[-1]:]

        # Solving TOV equations
        eps_of_n = sn.interpolate.PchipInterpolator(nn, EE)
        eps_of_P = sn.interpolate.PchipInterpolator(PP, EE)
        P_of_n = sn.interpolate.PchipInterpolator(nn, PP)
        deps_dn = eps_of_n.derivative()(nn)
        dp_dn = P_of_n.derivative()(nn)
        cs2_grid = dp_dn / deps_dn
        cs2_of_P = sn.interpolate.PchipInterpolator(PP, cs2_grid)

        if np.any(cs2_grid <= 0.0):
            raise ValueError("Negative sound-speed squared.")

        p_surface = PP[0]
        def surface_event(r, y):
            return y[1] - p_surface 
        surface_event.terminal = True
        surface_event.direction = -1.0

        def tov_love_rhs(r, state): 

            mass, P, y = state 

            E = float(eps_of_P(P))

            dm_dr = 4*np.pi*r**2*E
            dP_dr = -(E + P) * (mass + 4*np.pi*r**3*P)/(r*(r-2*mass))

            dp_dE = float(cs2_of_P(P))


            one_minus_2m_r = 1.0 - 2.0 * mass / r
            F = (1.0 - 4.0 * np.pi * r**2 * (E - P)) / one_minus_2m_r
            Q = (
                4.0 * np.pi / one_minus_2m_r
                * (5.0 * E + 9.0 * P + (E + P) / dp_dE)
                - 6.0 / (r**2 * one_minus_2m_r)
                - 4.0 * (mass + 4.0 * np.pi * r**3 * P)**2
                / (r**4 * one_minus_2m_r**2)
            )

            dy_dr = -(y**2 + y * F + r**2 * Q) / r
            

            return [dm_dr, dP_dr, dy_dr]


        def solve_star_love(Pc):

            r0 = 1e-5 
            y0 = 2.0

            eps0 = float(eps_of_P(Pc))
            m0 = 4.0*np.pi/3.0 * eps0 * r0**3

            sol = sn.integrate.solve_ivp(
                tov_love_rhs,
                [r0,50.0],
                [m0,Pc,y0],
                method="DOP853",
                events=surface_event,
                rtol=1e-8,
                atol=(1e-10, 1e-14, 1e-9)
            )
            if len(sol.t_events[0]) == 0:
                raise ValueError("The integration did not reach the stellar surface.")

            R = sol.t_events[0][0]

            M = sol.y_events[0][0][0]

            yR = sol.y_events[0][0][2]

            return M,R,yR


        masses = []
        lambdas = []

        M_SUN_KM = 1.4766250385

        RHO_VEC = np.geomspace(np.max([0.8*self.rho_0, self.n_min]), np.min([8*self.rho_0, self.n_max]),100)
        for Rho_c in RHO_VEC:
            try:
                M,R,yR = solve_star_love(P_of_n(Rho_c))

                masses.append(M/M_SUN_KM)
            
                lambdas.append(self.tidal_lambda(M,R,yR))
            except (ValueError, FloatingPointError, OverflowError):
                continue

        if len(masses) < 10:
            raise TypeError("Too few valid stellar models. Verify EOS parameters.")


        masses = np.asarray(masses)
        lambdas = np.asarray(lambdas)

        # Keep only the stable branch
        imax = np.argmax(masses)
        m1 = masses[:imax + 1]
        l1 = lambdas[:imax + 1]

        imin = np.argmin(m1)
        m2 = m1[imin:]
        l2 = l1[imin:]

        # Verify if Lambda(Masse) is monotonic for the interpolation, 
        # when it is not, usually at low masses, keep only the stable branch
        dx = np.diff(m2)
        turning_points = np.where(np.sign(dx[:-1]) != np.sign(dx[1:]))[0] + 1

        if len(turning_points) == 0:
            # Keep only possitive value of lambdas
            self.mass_table = m2[np.where(l2>= 0)[0]]
            self.Lambda_table = l2[np.where(l2>= 0)[0]]
        else:
            m3 = m2[np.max(turning_points):]
            l3 = l2[np.max(turning_points):]  
            self.mass_table = m3[np.where(l3>= 0)[0]]
            self.Lambda_table = l3 [np.where(l3>= 0)[0]]

        # Verfiy the value of squared sound speed in the density range of the stable branch
        idx_rho = np.where(nn <= RHO_VEC[imax])[0]
        if np.any(cs2_grid[idx_rho] > 1.0):
            raise ValueError("Acausal EOS.")
            

    def love_number(self, M, R, yR):

        C = M/R
        
        k2 = 8*C**5/5*(1-2*C)**2 * (2 + 2*C*(yR-1) - yR) / (2*C*(6 - 3*yR + 3*C*(5*yR-8))
            + 4*C**3*(13 - 11*yR + C*(3*yR - 2) + 2*C**2*(1+yR))
            + 3*(1-2*C)**2 * (2 - yR + 2*C*(yR-1)) * np.log(1-2*C))
        
        return k2


    def tidal_lambda(self, M, R, yR):

        k2 = self.love_number(M,R,yR)

        C = M/R

        Lambda = (2.0/3.0)*k2/C**5

        return Lambda


    def EOS_grid(self):

        # CORE: nuclear physics
        n_core = np.geomspace(self.n_t, self.n_t_1, 2000)

        try:
            delta_core = self.delta_beta_equilibrium(n_core)
        except (ValueError):
            raise ValueError('Check EOS Parameters.')
        
        e_core = self.energy_nucleon(n_core, delta_core)
        eps_core = self.energy_density_from_e(n_core, e_core)

        eps_n = sn.interpolate.PchipInterpolator(n_core, eps_core)
        deps_dn = eps_n.derivative()(n_core)
        P_core = n_core * deps_dn - eps_core



        # CRUST: polytropic
        n_crust = np.geomspace(self.n_min, self.n_t, 2000)

        # Determine K such as the EOS is continue
        P_t = P_core[0]
        K_crust = P_t / self.n_t**self.Gamma_crust

        P_crust = K_crust*n_crust**self.Gamma_crust

        eps_t = eps_core[0]
        A_crust = (eps_t - P_t / (self.Gamma_crust - 1.0)) / self.n_t
        eps_crust = (A_crust * n_crust + P_crust / (self.Gamma_crust - 1.0))


        # POLYTROPIC AT HIGH DENSITY
        n_1 = np.geomspace(self.n_t_1, self.n_t_2, 2000)

        P_t_1 = P_core[-1]
        K_1 = P_t_1 / self.n_t_1**self.Gamma_1

        P_1 = K_1*n_1**self.Gamma_1

        eps_t_1 = eps_core[-1]
        A_1 = (eps_t_1 - P_t_1 / (self.Gamma_1 - 1.0)) / self.n_t_1
        eps_1 = (A_1 * n_1 + P_1 / (self.Gamma_1 - 1.0))


        n_2 = np.geomspace(self.n_t_2, self.n_t_3, 2000)

        P_t_2 = P_1[-1]
        K_2 = P_t_2 / self.n_t_2**self.Gamma_2

        P_2 = K_2*n_2**self.Gamma_2

        eps_t_2 = eps_1[-1]
        A_2 = (eps_t_2 - P_t_2 / (self.Gamma_2 - 1.0)) / self.n_t_2
        eps_2 = (A_2 * n_2 + P_2 / (self.Gamma_2 - 1.0))


        n_3 = np.geomspace(self.n_t_3, self.n_max, 2000)

        P_t_3 = P_2[-1]
        K_3 = P_t_3 / self.n_t_3**self.Gamma_3

        P_3 = K_3*n_3**self.Gamma_3

        eps_t_3 = eps_2[-1]
        A_3 = (eps_t_3 - P_t_3 / (self.Gamma_3 - 1.0)) / self.n_t_3
        eps_3 = (A_3 * n_3 + P_3 / (self.Gamma_3 - 1.0))


        # Unit conversion
        G_SI = 6.67430e-11
        C_SI = 299_792_458.0
        MEVFM3_TO_GEOM = 1.602176634e32 * G_SI / C_SI**4 * 1.0e6

        n_table = np.concatenate([n_crust[:-1], n_core[:-1], n_1[:-1], n_2[:-1], n_3])
        P_table = np.concatenate([P_crust[:-1], P_core[:-1], P_1[:-1], P_2[:-1], P_3]) * MEVFM3_TO_GEOM
        eps_table = np.concatenate([eps_crust[:-1], eps_core[:-1], eps_1[:-1], eps_2[:-1], eps_3]) * MEVFM3_TO_GEOM

        return n_table, P_table, eps_table

   
    def delta_beta_equilibrium(self, rho):
        chi = (rho - self.rho_0)/(3*self.rho_0) 
        esym = self.e_sym_0 + self.L * chi + self.Ksym*0.5*chi**2

        if np.any(esym <= 0.0):
            raise ValueError("Symmetry energy becomes non-positive.")

        HBARC = 197.3269805             # MeV fm

        xi = (esym/HBARC)**2 * (24*rho*(1+np.sqrt(1+np.pi**2*rho*(HBARC/esym)**3/288)))**(1.0/3.0)
        Yp = 0.5 + (2*np.pi**2)**(1.0/3.0)/32*rho/xi*((2*np.pi**2)**(1.0/3.0)-xi**2/rho*(HBARC/esym)**3)

        delta = 1 - 2 * Yp

        return delta
    

    def energy_nucleon(self, rho, delta):
        chi = (rho - self.rho_0)/(3*self.rho_0)
        
        e0 = self.e0_rho0 + self.K0*0.5*chi**2.  
        esym = self.e_sym_0 + self.L * chi + self.Ksym*0.5*chi**2

        return e0 + esym * delta**2


    def energy_density_from_e(self, rho, e):
        m_N = 939.565 # Mev/c^2
        c = 1.
        return rho * (e + m_N *c**2)




    def EOS_L2ms(self,Lambda):
        
        Lambda_mass_relation = sn.interpolate.PchipInterpolator(self.Lambda_table, self.mass_table, extrapolate=False)
        msource = Lambda_mass_relation(Lambda)

        return msource
    
    def EOS_ms2L(self,msource):
        xp = get_module_array(msource)

        Lambda = xp.full_like(msource, np.nan, dtype=float)
        valid = (msource >= xp.min(self.mass_table)) & (msource <= xp.max(self.mass_table))

        Lambda_mass_relation = sn.interpolate.PchipInterpolator(self.mass_table, self.Lambda_table, extrapolate=False)
        Lambda[valid] = Lambda_mass_relation(msource[valid])

        return Lambda
    
    
    def log_pdf(self,lambda_1,lambda_2, msource1, msource2):
        xp = get_module_array(lambda_1)
        
        Lambda_from_mass_1 = self.EOS_ms2L(msource1)
        Lambda_from_mass_2 = self.EOS_ms2L(msource2)

        gauss_1 = -(xp.log(Lambda_from_mass_1)-xp.log(lambda_1))**2/(2*self.sigma_lambda**2) - xp.log(xp.sqrt(2*xp.pi)*self.sigma_lambda) - xp.log(lambda_1)
        gauss_2 = -(xp.log(Lambda_from_mass_2)-xp.log(lambda_2))**2/(2*self.sigma_lambda**2) - xp.log(xp.sqrt(2*xp.pi)*self.sigma_lambda) - xp.log(lambda_2)
   
        return gauss_1 + gauss_2 
        

    def pdf(self,lambda_1,lambda_2, msource1, msource2):
        xp = get_module_array(lambda_1)
        return xp.exp(self.log_pdf(lambda_1,lambda_2, msource1, msource2))




