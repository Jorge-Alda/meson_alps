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
BR_BKmumu.min_ma = 0
BR_BKmumu.max_ma = mB - mK - 2*mmu

def BR_B0Kstmumu(ma, couplings, f_a=1000):
    return BR_B0Ksta(ma, couplings, f_a=1000) * BR_amu(ma, couplings, f_a=1000)
BR_B0Kstmumu.tex = r'$\mathrm{BR}(B^0\to K^{*0} \mu^+\mu^-)$'
BR_B0Kstmumu.process_tex = r'$B^0\to K^{*0} \mu^+\mu^-$'
BR_B0Kstmumu.min_ma = 2*mmu
BR_B0Kstmumu.max_ma = mB0 - mKst0 - 2* mmu