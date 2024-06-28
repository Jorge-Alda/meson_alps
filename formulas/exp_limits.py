import pandas as pd
import numpy as np
from scipy.interpolate import interp1d, SmoothBivariateSpline
import os
dirname = os.path.dirname(__file__)
from .meson_leptonic import BR_Bs_int, BR_Bs_quad
from .constants import *
from .alp_decays import Gamma_a_tot
import flavio

lim_BKinv = 1.2e-5
lim_Kpiee = 3.18e-7
lim_Kpigammagamma = 1.01e-7+2* 0.06e-7

lim_KLmumu = 6.84e-9 + 2* 0.11e-9

df_e949 = pd.read_csv(os.path.join(dirname, '../data/E949data.csv'), names=['ma', 'br'])
e949_interp = interp1d(df_e949['ma']/1000, df_e949['br'], kind='cubic')
def lim_Kpiinv(ma):
    if ma < 0.115 or (ma > 0.150 and ma < 0.260):
        return float(e949_interp(ma))
    else: 
        return float('nan')


# Old data, just for comparison with Olcyr. Later we'll update the values
central_Bsmumu = 3e-9
error_Bsmumu = (0.6e-9**2+0.3e-9**2)**0.5
BR_SM_Bsmumu = flavio.sm_prediction('BR(Bs->mumu)')
def lim_Bsmumu(ma, couplings, f_a=1000):
    br_quad = BR_Bs_quad(ma, mmu, couplings, f_a)
    br_int = BR_Bs_int(ma, mmu, couplings, f_a)
    delta = BR_SM_Bsmumu - (central_Bsmumu + 2*error_Bsmumu)
    return np.sqrt((-br_int+np.sqrt(br_int**2-4*br_quad*delta))/(2*br_quad))

lim_KLmumu = 6.84e-9+2*0.11e-9

df_BKmumu_LHCb = pd.read_csv(os.path.join(dirname, '../data/BKmumu_LHCb.csv'))
br_LHCb_interp = SmoothBivariateSpline(df_BKmumu_LHCb['ma_GeV'], df_BKmumu_LHCb['logtau_ps'], df_BKmumu_LHCb['logbr'])
def lim_BKmumu_LHCb(ma, couplings, f_a=1000):   
    boundaries_ma = [0.2535545, # Starting mass
                     0.3886255924170616, # Start of vetoed KS0 region
                     0.48815165876777245, # End of vetoed KS0 region
                     0.9857819905213271, # Start of partially vetoed phi region
                     1.0710900473933649, # End of partially vetoed phi region
                     2.940758293838863, # Start of vetoed J/psi region
                     3.1895734597156395, # End of vetoed J/psi region
                     3.580568720379147, # Start of vetoed psi(2S) and psi(3770) region
                     3.85781990521327, # End of vetoed psi(2S) and psi(3770)
                     4.113744075829384, # Start of partially vetoed psi(4160) region
                     4.270142180094787, # End of partially vetoed psi(4160) region
                     4.69668246] # Final mass
    logtaures = -0.81027668

    if ma < boundaries_ma[0]:
        return float('nan')
    if ma > boundaries_ma[1] and ma < boundaries_ma[2]:
        return float('nan')
    if ma > boundaries_ma[5] and ma < boundaries_ma[6]:
        return float('nan')
    if ma > boundaries_ma[7] and ma < boundaries_ma[8]:
        return float('nan')
    if ma > boundaries_ma[11]:
        return float('nan')

    logtau = np.log10(hbar_GeVps/Gamma_a_tot(ma, couplings, f_a))
    if logtau < -1:
        return float('nan')
    if logtau > 3:
        return float('nan')

    if ma > boundaries_ma[3] and ma < boundaries_ma[4] and logtau < logtaures:
        return float('nan')
    if ma > boundaries_ma[9] and ma < boundaries_ma[10] and logtau < logtaures:
        return float('nan')

    return 10**float(br_LHCb_interp(ma, logtau))