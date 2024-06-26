import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import os
dirname = os.path.dirname(__file__)
from .meson_leptonic import BR_Bs_int, BR_Bs_quad
from .constants import *
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