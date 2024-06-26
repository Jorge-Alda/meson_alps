import numpy as np
from .couplings import gq_eff
from .common import f0_BK, f0_Kpi, kallen
from .alp_decays import BR_ae, BR_agamma, BR_amu
from .constants import *

def BR_Kpia(ma, couplings, f_a=1000):
    return mK**3*abs(gq_eff('sd', couplings, f_a))**2/(64*np.pi) * f0_Kpi(ma**2)**2*np.sqrt(kallen(1, mpi_pm**2/mK**2, ma**2/mK**2))*(1-mpi_pm**2/mK**2)**2/GammaK

def BR_Kpimumu(ma, couplings, f_a=1000):
    return BR_Kpia(ma, couplings, f_a=1000) * BR_amu(ma, couplings, f_a=1000)

def BR_Kpiee(ma, couplings, f_a=1000):
    return BR_Kpia(ma, couplings, f_a=1000) * BR_ae(ma, couplings, f_a=1000)

def BR_Kpigamma(ma, couplings, f_a=1000):
    return BR_Kpia(ma, couplings, f_a=1000) * BR_agamma(ma, couplings, f_a=1000)

def BR_BKa(ma, couplings, f_a=1000):
    return mB**3*abs(gq_eff('bs', couplings, f_a))**2/(64*np.pi) * f0_BK(ma**2)**2*np.sqrt(kallen(1, mK**2/mB**2, ma**2/mB**2))*(1-mK**2/mB**2)**2/GammaB

def BR_BKmumu(ma, couplings, f_a=1000):
    return BR_BKa(ma, couplings, f_a=1000) * BR_amu(ma, couplings, f_a=1000)