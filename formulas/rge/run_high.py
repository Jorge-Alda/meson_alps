"""Running of the RG for the ALP couplings above the EW scale

Auxiliary functions
-------------------

    gauge_tilde
    sm_params

Beta functions
--------------
    beta_ytop


Running
-------
    run_leadinglog
"""

import numpy as np
from . import ALPcouplings
import wilson
from typing import Callable
from scipy.integrate import solve_ivp


def gauge_tilde(couplings: ALPcouplings) -> dict:
    """Calculate the gauge effective couplings invariant under filed redefinitions of the fermions
    
    Implements eq.(20) of 2012.12272

    Parameters
    ----------
    couplings : ALPCouplings
        Object containing the ALP couplings

    Returns
    -------
    cXtilde : dict
        Dictionary containing `cgtilde`, `cBtilde` and `cWtilde`
    """
    couplings = couplings.translate('derivative_above')
    cg = couplings.values['cg'] - 0.5 *np.trace(2*couplings.values['cqL']-couplings.values['cuR']-couplings.values['cdR'])
    cW = couplings.values['cW'] - 0.5*np.trace(3*couplings.values['cqL']+ couplings.values['clL'])
    cB = couplings.values['cB']-np.trace(couplings.values['cqL']-8*couplings.values['cuR']-2*couplings.values['cdR']+3*couplings.values['clL']-6*couplings.values['ceR'])/6
    return {'cgtilde': cg, 'cBtilde':cB, 'cWtilde': cW}


def sm_params(scale: float) -> dict:
    """SM parameters at an energy scale
    
    Parameters
    ----------
    scale : float
        Energy scale, in GeV

    Returns
    -------
    pars : dict
        dict containing the Yukawa matrices `yu`, `yd` and `ye`, and the gauge couplings `alpha_s`, `alpha_1` and `alpha_2`
    """

    wSM = wilson.classes.SMEFT(wilson.wcxf.WC('SMEFT', 'Warsaw', scale, {})).C_in # For the moment we reuse wilson's code for the SM case, i.e, with all Wilson coefficients set to zero. Maybe at some point we should implement our own version.
    return {
        'yu': np.matrix(wSM['Gu']),
        'yd': np.matrix(wSM['Gd']),
        'ye': np.matrix(wSM['Ge']),
        'alpha_s': wSM['gs']**2/(4*np.pi),
        'alpha_1': wSM['gp']**2/(4*np.pi),
        'alpha_2': wSM['g']**2/(4*np.pi)
    }


def beta_ytop(couplings: ALPcouplings) -> ALPcouplings:
    """beta function for the ALP couplings, neglecting all Yukawas except y_top
    
    Implements eq.(24) of 2012.12272

    Parameters
    ----------
    couplings : ALPCouplings
        An object containing the ALP couplings

    Returns
    -------
    beta : ALPCouplings
        An object containing the beta functions of the ALP couplings
    """

    tildes = gauge_tilde(couplings)
    pars = sm_params(couplings.scale)
    ytop = np.real(pars['yu'][2,2])
    alpha_s = pars['alpha_s']
    alpha_1 = pars['alpha_1']
    alpha_2 = pars['alpha_2']

    # Field redefinitions of the fermionic fields that eliminate Ophi, see Eq.(5) of 2012.12272 and the discussion below
    bu = -1
    bd = 1
    be = 1
    bQ = 0
    bL = 0

    # eq(25)
    couplings = couplings.translate('derivative_above')
    ctt = couplings.values['cuR'][2,2]-couplings.values['cqL'][2,2]

    # eq(24)
    diag_betaqL = -2*ytop**2*np.matrix(np.diag([0,0,1]) + 3*bQ*np.eye(3))*ctt +np.eye(3)*(-16*alpha_s**2*tildes['cgtilde'] - 9*alpha_2**2*tildes['cWtilde'] - 1/3*alpha_1**2*tildes['cBtilde'])
    offdiag_betaqL = np.matrix(np.zeros([3,3]))
    offdiag_betaqL[0,2] = couplings.values['cqL'][0,2]
    offdiag_betaqL[1,2] = couplings.values['cqL'][1,2]
    offdiag_betaqL[2,0] = couplings.values['cqL'][2,0]
    offdiag_betaqL[2,1] = couplings.values['cqL'][2,1]
    betaqL = diag_betaqL + ytop**2/2 * offdiag_betaqL

    diag_betauR = 2*ytop**2*np.matrix(np.diag([0,0,1])-3*bu*np.eye(3))*ctt + np.eye(3)*(16*alpha_s**2*tildes['cgtilde']+16/3*alpha_1**2*tildes['cBtilde'])
    offdiag_betauR = np.matrix(np.zeros([3,3]))
    offdiag_betauR[0,2] = couplings.values['cuR'][0,2]
    offdiag_betauR[1,2] = couplings.values['cuR'][1,2]
    offdiag_betauR[2,0] = couplings.values['cuR'][2,0]
    offdiag_betauR[2,1] = couplings.values['cuR'][2,1]
    betauR = diag_betauR + ytop**2*offdiag_betauR

    betadR = np.eye(3)*(-6*ytop**2*bd*ctt + 16*alpha_s**2*tildes['cgtilde']+16/12*alpha_1**2*tildes['cBtilde'])

    betalL = np.eye(3)*(-6*ytop**2*bL*ctt-9*alpha_2**2*tildes['cWtilde']-3*alpha_1**2*tildes['cBtilde'])

    betaeR = np.eye(3)*(-16*ytop**2*be*ctt+12*alpha_1**2*tildes['cBtilde'])

    return ALPcouplings({'cqL': betaqL, 'cuR': betauR, 'cdR': betadR, 'clL': betalL, 'ceR': betaeR}, couplings.scale, 'derivative_above')


def run_leadinglog(couplings: ALPcouplings, beta: Callable[[ALPcouplings], ALPcouplings], scale_out: float) -> ALPcouplings:
    """Obtain the ALP couplings at a different scale using the leading log approximation
    
    Parameters
    ----------
    couplings : ALPcouplings
        Object containing the ALP couplings at the original scale

    beta : Callable[ALPcouplings, ALPcouplings]
        Function that return the beta function

    scale_out : float
        Final energy scale, in GeV
    """

    result = couplings + beta(couplings) * (np.log(scale_out/couplings.scale)/(16*np.pi**2))
    result.scale = scale_out
    return result


def run_scipy(couplings: ALPcouplings, beta: Callable[[ALPcouplings], ALPcouplings], scale_out: float) -> ALPcouplings:
    """Obtain the ALP couplings at a different scale using scipy's integration
    
    Parameters
    ----------
    couplings : ALPcouplings
        Object containing the ALP couplings at the original scale

    beta : Callable[ALPcouplings, ALPcouplings]
        Function that return the beta function

    scale_out : float
        Final energy scale, in GeV
    """

    def fun(t0, y):
        return beta(ALPcouplings.fromarray(y, np.exp(t0), 'derivative_above')).toarray()/(16*np.pi**2)
    
    sol = solve_ivp(fun=fun, t_span=(np.log(couplings.scale), np.log(scale_out)), y0=couplings.translate('derivative_above').toarray())
    return sol