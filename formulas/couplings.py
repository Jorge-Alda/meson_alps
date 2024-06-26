from collections import defaultdict
import numpy as np
import flavio
from particle import Particle
from functools import lru_cache
pars = flavio.default_parameters.get_central_all()
from .common import B1, B2, alpha_em
from .constants import *

#####################################
# Effective FCNC quark-ALP coupling #
#####################################

@lru_cache
def gq_eff_caPhi(transition, f_a=1000):
    tot = 0
    if transition[0] in ['b', 's', 'd'] and transition[0] in ['b', 's', 'd']:
        quarks = ['u', 'c', 't']
    elif transition[0] in ['t', 'c', 'u'] and transition[0] in ['t', 'c', 'u']:
        quarks = ['d', 's', 'b']
    else:
        raise ValueError(f'Unknown quark transition {transition}')

    for q in quarks:
        mq = mq_dict[q]
        ckm = flavio.physics.ckm.xi(q, transition)(pars)
        tot += ckm*mq**2/mW**2*np.log(f_a**2/mq**2)
       
    return -0.25*g2**2/(16*np.pi**2*f_a)*tot

@lru_cache
def gq_eff_cW(transition, f_a=1000):
    tot = 0
    if transition[0] in ['b', 's', 'd'] and transition[0] in ['b', 's', 'd']:
        quarks = ['u', 'c', 't']
    elif transition[0] in ['t', 'c', 'u'] and transition[0] in ['t', 'c', 'u']:
        quarks = ['d', 's', 'b']
    else:
        raise ValueError(f'Unknown quark transition {transition}')
    
    def gloop(mquark):
        x = mquark**2/mW**2
        return x*(1+x*np.log(x)-x)/(1-x)**2

    for q in quarks:
        mq = mq_dict[q]
        ckm = flavio.physics.ckm.xi(q, transition)(pars)
        tot += ckm*gloop(mq)
        
    return 3*g2**2/(16*np.pi**2*f_a)*tot

def gq_eff(transition, couplings, f_a=1000):
    c = 0
    if 'caPhi' in couplings.keys():
        c += couplings['caPhi'] * gq_eff_caPhi(transition, f_a)
    if 'cW' in couplings.keys():
        c += couplings['cW'] * gq_eff_cW(transition, f_a)
    return c

#####################################
#   Effective photon-ALP coupling   #
#####################################

def cgamma_eff_caPhi(ma):
    light_quarks = int(ma>1)
    B0 = 3*4/9*(B1(4*mt**2/ma**2)+B1(4*mc**2/ma**2)+light_quarks*B1(4*mu**2/ma**2)) - 3/9*(B1(4*mb**2/ma**2)+light_quarks*B1(4*ms**2/ma**2)+light_quarks*B1(4*md**2/ma**2)) - (B1(4*mtau**2/ma**2)+B1(4*mmu**2/ma**2)+B1(4*me**2/ma**2))
    return -0.25*alpha_em(ma**2)/np.pi*(-(1-light_quarks)*ma**2/(mpi_pm**2-ma**2)+B0)

def cgamma_eff_cW(ma):
    def floop(x):
        if x >= 1:
            return np.arcsin(x**(-0.5))
        else:
            return np.pi/2+0.5j*np.log((1+np.sqrt(1-x))/(1-np.sqrt(1-x)))
            
    x = 4*mW**2/ma**2
    return s2w + 2*alpha_em(mW**2)/np.pi*B2(x)

def cgamma_eff(ma, couplings):
    c = 0
    if 'caPhi' in couplings.keys():
        c += couplings['caPhi'] * cgamma_eff_caPhi(ma)
    if 'cW' in couplings.keys():
        c += couplings['cW'] * cgamma_eff_cW(ma)
    if 'cB' in couplings.keys():
        c += couplings['cB']*(1-s2w)
    return c

#####################################
#   Effective fermion-ALP coupling  #
#####################################

@lru_cache
def cll_eff_cW(m_ferm, f_a=1000):
    return alpha_em(m_ferm**2)/np.pi*(9/4*np.log(f_a/mW)/s2w + 6*s2w*np.log(mW/m_ferm))

@lru_cache
def cll_eff_cB(m_ferm, f_a=1000):
    return alpha_em(m_ferm**2)/np.pi*(15/4*np.log(f_a/mW)/(1-s2w) + 6*(1-s2w)*np.log(mW/m_ferm))

def cll_eff(m_ferm, couplings, f_a=1000):
    c = 0
    if 'caPhi' in couplings.keys():
        c += couplings['caPhi']
    if 'cW' in couplings.keys():
        c += couplings['cW'] * cll_eff_cW(m_ferm, f_a)
    if 'cB' in couplings.keys():
        c += couplings['cB']*cll_eff_cB(m_ferm, f_a)
    return c


#####################################
#   Effective gluon-ALP coupling    #
#####################################

def cgg_eff(ma, couplings):
    ceff = 0
    for q in ['u', 'd', 's', 'c', 'b']:
        mq = mq_dict[q]
        ceff += 0.5 * cll_eff(mq, couplings) * B1(4*mq**2/ma**2)
    return ceff

#####################################
#    Effective pion-ALP coupling    #
#####################################

def cpi_eff(ma, couplings):
    return -2*cgg_eff(ma, couplings)*(md-mu)/(md+mu)-cll_eff(mu, couplings)+cll_eff(md, couplings)

@lru_cache
def gSM(transition):
    tot = 0
    if transition[0] in ['b', 's', 'd'] and transition[0] in ['b', 's', 'd']:
        quarks = ['u', 'c', 't']
    elif transition[0] in ['t', 'c', 'u'] and transition[0] in ['t', 'c', 'u']:
        quarks = ['d', 's', 'b']
    else:
        raise ValueError(f'Unknown quark transition {transition}')

    for q in quarks:
        mq = mq_dict[q]
        xq = mq**2/mW**2
        ckm = flavio.physics.ckm.xi(q, transition)(pars)
        tot += 0.125*ckm*xq*((xq-6)/(xq-1)+(3*xq+2)/(xq-1)**2*np.log(xq))
       
    return tot