import numpy as np

from ..common import f0_BK, f0_Kpi, A0_BKst, kallen
from ..couplings import gq_eff
from ..constants import *

def BR_Kpia(ma, couplings, f_a=1000):
    return mK**3*abs(gq_eff('sd', couplings, f_a))**2/(64*np.pi) * f0_Kpi(ma**2)**2*np.sqrt(kallen(1, mpi_pm**2/mK**2, ma**2/mK**2))*(1-mpi_pm**2/mK**2)**2/GammaK
BR_Kpia.tex = r'$\mathrm{BR}(K^\pm\to\pi^\pm a)$'
BR_Kpia.process_tex = r'$K^\pm\to\pi^\pm a$'
BR_Kpia.min_ma = 0
BR_Kpia.max_ma = mK - mpi_pm

def BR_BKa(ma, couplings, f_a=1000):
    return mB**3*abs(gq_eff('bs', couplings, f_a))**2/(64*np.pi) * f0_BK(ma**2)**2*np.sqrt(kallen(1, mK**2/mB**2, ma**2/mB**2))*(1-mK**2/mB**2)**2/GammaB
BR_BKa.tex = r'$\mathrm{BR}(B^\pm\to K^\pm a)$'
BR_BKa.process_tex = r'$B^\pm\to K^\pm a$'
BR_BKa.min_ma = 0
BR_BKa.max_ma = mB - mK

def BR_B0Ksta(ma, couplings, f_a=1000):
    return mB0**3*abs(gq_eff('bs', couplings, f_a))**2/(64*np.pi) * A0_BKst(ma**2)**2 * kallen(1, mKst0**2/mB0**2, ma**2/mB0**2)**1.5/GammaB0
BR_B0Ksta.tex = r'$\mathrm{BR}(B^0\to K^{*0} a)$'
BR_B0Ksta.process_tex = r'$B^0\to K^{*0} a$'
BR_B0Ksta.min_ma = 0
BR_B0Ksta.max_ma = mB0 - mKst0