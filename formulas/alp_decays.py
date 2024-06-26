import numpy as np
from scipy.integrate import quad
from .common import B1, kallen
from .constants import *
from .couplings import *

#############
# a -> 3 pi #
#############

def Gamma_3pi0(ma, couplings, f_a=1000):
    g00 = lambda r: 2/(1-r)**2*quad(lambda z: np.sqrt(1-4*r/z)*np.sqrt(kallen(1, z, r)), 4*r, (1-np.sqrt(r))**2)[0]
    #ceff = 1/(32*np.pi**2)*(B1(4*mu**2/ma**2)+B1(4*md**2/ma**2)+B1(4*ms**2/ma**2))
    #return np.pi/6 * ma*mpi**4/(f_a**2*fpi**2)*(ceff*(md-mu)/(md+mu))**2*g00(mpi**2/ma**2)
    ceff = 0.5*(B1(4*mu**2/ma**2)+B1(4*md**2/ma**2)+B1(4*ms**2/ma**2))
    return ma*mpi_pm**4/(6144*np.pi**3*fpi**2*f_a**2)*np.abs(cpi_eff(ma, couplings))**2*g00(mpi_pm**2/ma**2)

def Gamma_3pipm(ma, couplings, f_a=1000):
    gpm = lambda r: 12/(1-r)**2*quad(lambda z: np.sqrt(1-4*r/z)*np.sqrt(kallen(1, z, r))*(z-r)**2, 4*r, (1-np.sqrt(r))**2)[0]
    ceff = 0.5*(B1(4*mu**2/ma**2)+B1(4*md**2/ma**2)+B1(4*ms**2/ma**2))
    return ma*mpi_pm**4/(6144*np.pi**3*fpi**2*f_a**2)*np.abs(cpi_eff(ma, couplings))**2*gpm(mpi_pm**2/ma**2)

#################
# a -> fermions #
#################

def Gamma_aferm(ma, mferm, couplings, f_a=1000):
    return np.abs(cll_eff(mferm, couplings, f_a=1000))**2*ma*mferm**2/(8*np.pi*f_a**2)*np.sqrt(1-4*mferm**2/ma**2)

def Gamma_aheavyq(ma, couplings, f_a=1000):
    if ma > 1 and ma < 3:
        return 3*Gamma_aferm(3, mc, couplings, f_a=1000)
    elif ma < 2*mb:
        return 3*Gamma_aferm(ma, mc, couplings, f_a=1000)
    else:
        return 3*Gamma_aferm(ma, mc, couplings, f_a=1000)+3*Gamma_aferm(ma, mb, couplings, f_a=1000)

#################
# a -> photons  #
#################

def Gamma_agamma(ma, couplings, f_a=1000):
    return abs(cgamma_eff(ma, couplings))**2*ma**3/(4*np.pi*f_a**2)

#################
#   a -> total  #
#################

def Gamma_a_tot(ma, couplings, f_a=1000):
    Gamma = Gamma_agamma(ma, couplings, f_a=1000)
    if ma > 2*me:
        Gamma += Gamma_aferm(ma, me, couplings, f_a=1000)
    if ma > 2*mmu:
        Gamma += Gamma_aferm(ma, mmu, couplings, f_a=1000)
    if ma > 2*mtau:
        Gamma += Gamma_aferm(ma, mtau, couplings, f_a=1000)
    if ma > 1:
        Gamma += Gamma_aheavyq(ma, couplings, f_a=1000)
    if ma > 3*mpi_pm:
        Gamma += Gamma_3pi0(ma, couplings, f_a=1000) + Gamma_3pipm(ma, couplings, f_a=1000)
    return Gamma

#######
# BR #
######

def BR_agamma(ma, couplings, f_a=1000):
    return Gamma_agamma(ma, couplings, f_a=1000)/Gamma_a_tot(ma, couplings, f_a=1000)

def BR_ae(ma, couplings, f_a=1000):
    if ma < 2*me:
        return 0
    return Gamma_aferm(ma, me, couplings, f_a=1000)/Gamma_a_tot(ma, couplings, f_a=1000)

def BR_amu(ma, couplings, f_a=1000):
    if ma < 2*mmu:
        return 0
    return Gamma_aferm(ma, mmu, couplings, f_a=1000)/Gamma_a_tot(ma, couplings, f_a=1000)