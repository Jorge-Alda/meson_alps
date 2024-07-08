import numpy as np

from .longlived import *
from ..alp_decays import BR_ae, BR_agamma, BR_amu
from ..constants import *

def BR_Kpimumu(ma, couplings, f_a=1000):
    return BR_Kpia(ma, couplings, f_a=1000) * BR_amu(ma, couplings, f_a=1000)
BR_Kpimumu.tex = r'$\mathrm{BR}(K^\pm\to\pi^\pm \mu^+\mu^-)$'
BR_Kpimumu.process_tex = r'$K^\pm\to\pi^\pm \mu^+\mu^-$'
BR_Kpimumu.min_mass = 2*mmu
BR_Kpimumu.max_mass = mK - mpi_pm - 2*mmu

def BR_Kpiee(ma, couplings, f_a=1000):
    return BR_Kpia(ma, couplings, f_a=1000) * BR_ae(ma, couplings, f_a=1000)
BR_Kpiee.tex = r'$\mathrm{BR}(K^\pm\to\pi^\pm e^+e^-)$'
BR_Kpiee.process_tex = r'$K^\pm\to\pi^\pm e^+e^-$'
BR_Kpiee.min_mass = 2*me
BR_Kpiee.max_mass = mK - mpi_pm - 2*me

def BR_Kpigamma(ma, couplings, f_a=1000):
    return BR_Kpia(ma, couplings, f_a=1000) * BR_agamma(ma, couplings, f_a=1000)
BR_Kpigamma.tex = r'$\mathrm{BR}(K^\pm\to\pi^\pm \gamma\gamma)$'
BR_Kpigamma.process_tex = r'$K^\pm\to\pi^\pm \gamma\gamma$'
BR_Kpigamma.min_ma = 0
BR_Kpigamma.max_ma = mK - mpi_pm

def BR_BKmumu(ma, couplings, f_a=1000):
    return BR_BKa(ma, couplings, f_a=1000) * BR_amu(ma, couplings, f_a=1000)
BR_BKmumu.tex = r'$\mathrm{BR}(B^\pm\to K^\pm \mu^+\mu^-)$'
BR_BKmumu.process_tex = r'$B^\pm\to K^\pm \mu^+\mu^-$'
BR_BKmumu.min_ma = 2*mmu
BR_BKmumu.max_ma = mB - mK - 2*mmu

def BR_B0Kstmumu(ma, couplings, f_a=1000):
    return BR_B0Ksta(ma, couplings, f_a=1000) * BR_amu(ma, couplings, f_a=1000)
BR_B0Kstmumu.tex = r'$\mathrm{BR}(B^0\to K^{*0} \mu^+\mu^-)$'
BR_B0Kstmumu.process_tex = r'$B^0\to K^{*0} \mu^+\mu^-$'
BR_B0Kstmumu.min_ma = 2*mmu
BR_B0Kstmumu.max_ma = mB0 - mKst0 - 2* mmu

def BR_BKmumu_offshell(ma, couplings, q2min, q2max, f_a=1000):
    if ma**2 > q2min and ma**2 < q2max:
        return BR_BKmumu(ma, couplings, f_a)
    integrand = lambda q2: q2*np.sqrt(1-4*mmu**2/q2)*np.sqrt(kallen(mB**2, mK**2, q2))/np.abs(q2-ma**2)**2*f0_BK(q2)**2
    integral = quad(integrand, q2min, q2max)[0]
    return integral*np.abs(gq_eff('bs', couplings, f_a) * cll_eff(mmu, couplings, f_a=1000))**2*mmu**2/f_a**2*(mB**2-mK**2)**2/((2*np.pi)**3*32*mB**3*GammaB)
BR_BKmumu_offshell.tex = r'$\mathrm{BR}(B^\pm\to K^\pm \mu^+\mu^-)$'
BR_BKmumu_offshell.process_tex = r'$B^\pm\to K^\pm \mu^+\mu^-$'
BR_BKmumu_offshell.min_ma = 0
BR_BKmumu_offshell.max_ma = np.inf