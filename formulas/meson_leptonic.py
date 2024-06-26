from .couplings import gq_eff, gSM, cll_eff
from .common import *
from .constants import *
from .alp_decays import Gamma_a_tot


def BR_B0_quad(ma, mlep, couplings, f_a=1000):
    # Correction due to meson oscillations: 1204.1737
    yd = pars['DeltaGamma/Gamma_B0']/2.
    corr_tauBd = 1/(1-yd)
    return np.abs(gq_eff('bd', couplings, f_a))**2/(16*np.pi) * np.abs(cll_eff(mlep, couplings, f_a))**2/f_a**2 * mlep**2 * mB0 * fB**2 * (mB0**2/(mB0**2-ma**2))**2 * np.sqrt(1-4*mlep**2/mB0**2)/GammaB0*corr_tauBd
    
def BR_B0_int(ma, mlep, couplings, f_a=1000):
    yd = pars['DeltaGamma/Gamma_B0']/2.
    corr_tauBd = 1/(1-yd)
    #return -alpha_em(ma**2)**2 *  np.real(gq_eff('bd', couplings, f_a) * gSM('bd') * cll_eff(mlep, couplings, f_a))/(2*np.pi**2*s2w**2*(1-s2w)**0.5*f_a*mW) * mlep**2 * mB0 * fB**2 * (mB0**2/(mB0**2-ma**2)) * np.sqrt(1-4*mlep**2/mB0**2)/GammaB0*corr_tauBd
    amp_SM = GF*alpha_em(mlep**2)/np.pi*mB0*mlep*fB*flavio.physics.ckm.xi('t', 'bd')(pars)*C10_SM # IMPORTANT, wrong C10 (should be C10_bd)
    amp_ALP = gq_eff('bd', couplings, f_a)*cll_eff(mlep, couplings, f_a)/f_a*fB*mlep*mB0**3/(mB0**2-ma**2)/2**0.5
    return 2*np.real(amp_SM*np.conjugate(amp_ALP))/(16*np.pi*mB0)* np.sqrt(1-4*mlep**2/mB0**2)/GammaB0*corr_tauBd

def BR_Bs_quad(ma, mlep, couplings, f_a=1000):
    # Correction due to meson oscillations: 1204.1737
    ys = pars['DeltaGamma/Gamma_Bs']/2.
    corr_tauBs = 1/(1-ys)
    return np.abs(gq_eff('bs', couplings, f_a))**2/(16*np.pi) * np.abs(cll_eff(mlep, couplings, f_a) * mBs**2/(mBs**2-ma**2+1j*ma*Gamma_a_tot(ma, couplings, f_a)))**2/f_a**2 * mlep**2 * mBs * fBs**2 * np.sqrt(1-4*mlep**2/mBs**2)/GammaBs*corr_tauBs

def BR_Bs_int(ma, mlep, couplings, f_a=1000):
    ys = pars['DeltaGamma/Gamma_Bs']/2.
    corr_tauBs = 1/(1-ys)
    #return -alpha_em(ma**2)**2 *  np.real(gq_eff('bs', couplings, f_a) * gSM('bs') * cll_eff(mlep, couplings, f_a))/(2*np.pi**2*s2w**2*(1-s2w)**0.5*f_a*mW) * mlep**2 * mBs * fB**2 * (mBs**2/(mBs**2-ma**2)) * np.sqrt(1-4*mlep**2/mBs**2)/GammaBs*corr_tauBs
    amp_SM = GF*alpha_em(mlep**2)/np.pi*mBs*mlep*fBs*flavio.physics.ckm.xi('t', 'bs')(pars)*C10_SM
    amp_ALP = gq_eff('bs', couplings, f_a)*cll_eff(mlep, couplings, f_a)/f_a*fBs*mlep*mBs**3/(mBs**2-ma**2+1j*ma*Gamma_a_tot(ma, couplings, f_a))/2**0.5
    return 2*np.real(amp_SM*np.conjugate(amp_ALP))/(16*np.pi*mBs)* np.sqrt(1-4*mlep**2/mBs**2)/GammaBs*corr_tauBs

def BR_KL_quad(ma, mlep, couplings, f_a=1000):
    return np.real(gq_eff('sd', couplings, f_a))**2/(16*np.pi) * np.abs(cll_eff(mlep, couplings, f_a))**2/f_a**2 * mlep**2 * mKL * fK**2 * (mKL**2/(mKL**2-ma**2))**2 * np.sqrt(1-4*mlep**2/mKL**2)/GammaKL

def BR_KL_int(ma, mlep, couplings, f_a=1000):
    return -alpha_em(ma**2)**2 *  np.real(gq_eff('sd', couplings, f_a)) * np.real(gSM('sd')) * np.real(cll_eff(mlep, couplings, f_a))/(2*np.pi**2*s2w**2*(1-s2w)**0.5*f_a*mW) * mlep**2 * mKL * fK**2 * (mKL**2/(mKL**2-ma**2)) * np.sqrt(1-4*mlep**2/mKL**2)/GammaKL

def BR_KL(ma, mlep, couplings, f_a=1000):
    return BR_KL_quad(ma, mlep, couplings, f_a=1000) + BR_KL_int(ma, mlep, couplings, f_a=1000)