import numpy as np
import wilson
from scipy.integrate import solve_ivp
from .constants import *
from .common import alpha_em

class Couplings:
    def __init__(self, values, scale, basis, eft='he-alp'):
        if (basis, eft) not in [('derivative', 'he-alp'), ('yukawa', 'he-alp'), ('bosonic', 'he-alp'), ('k', 'le-alp'), ('cf', 'le-alp')]:
            raise ValueError("Wrong basis/EFT")
        self.values = values
        self.scale = scale
        self.basis = basis
        self.eft = eft

def extract_cf(couplings, f):
    if f'c{f}' not in couplings.keys():
        c = np.matrix(np.zeros((3,3)))
    elif isinstance(couplings[f'c{f}'], (int, float, complex)):
        c = couplings[f'c{f}']*np.matrix(np.eye(3))
    else:
        c = couplings[f'c{f}']
    if 'cPhi' in couplings.keys():
        c += couplings['cPhi']*np.eye(3)
    return c

def dict2column(couplings):
    v = []
    for f in ['qL', 'uR', 'dR', 'lL', 'eR']:
        m = extract_cf(couplings, f)
        s = np.zeros(6, dtype=complex)
        s[0] = m[0,0]
        s[1] = m[0,1]
        s[2] = m[0,2]
        s[3] = m[1,1]
        s[4] = m[1,2]
        s[5] = m[2,2]
        v.append(s)
    if 'cg' in couplings.keys():
        cg = couplings['cg']
    else:
        cg = 0
    if 'cW' in couplings.keys():
        cW = couplings['cW']
    else:
        cW = 0
    if 'cB' in couplings.keys():
        cB = couplings['cB']
    else:
        cB = 0
    v.append([cg, cW, cB])
    return np.concatenate(v)

def column2dict(v):
    d = {}
    for i, f in enumerate(['cqL', 'cuR', 'cdR', 'clL', 'ceR']):
        m = np.matrix(np.zeros((3,3), dtype=complex))
        m[0,0] = v[6*i]
        m[0,1] = v[6*i+1]
        m[1,0] = np.conjugate(v[6*i+1])
        m[0,2] = v[6*i+2]
        m[2,0] = np.conjugate(v[6*i+2])
        m[1,1] = v[6*i+3]
        m[1,2] = v[6*i+4]
        m[2,1] = np.conjugate(v[6*i+4])
        m[2,2] = v[6*i+5]
        d |= {f: m}
    d |= {'cg': v[30], 'cW': v[31], 'cB': v[32]}
    return d

def beta(mu0, vec):
    couplings = column2dict(vec)
    cqL = couplings['cqL']
    clL = couplings['clL']
    cdR = couplings['cdR']
    cuR = couplings['cuR']
    ceR = couplings['ceR']
    cg = couplings['cg']
    cW = couplings['cW']
    cB = couplings['cB']
    
    wSM = wilson.classes.SMEFT(wilson.wcxf.WC('SMEFT', 'Warsaw', mu0, {})).C_in
    yu = np.matrix(wSM['Gu'])
    yd = np.matrix(wSM['Gd'])
    ye = np.matrix(wSM['Ge'])
    alpha_s = wSM['gs']**2/(4*np.pi)
    alpha_1 = wSM['gp']**2/(4*np.pi)
    alpha_2 = wSM['g']**2/(4*np.pi)

    X = np.trace(3*cqL @ (yu @ yu.H - yd @ yd.H) - 3*cuR @ yu.H @ yu + 3*cdR @ yd.H @ yd - clL @ ye @ ye.H + ceR @ ye.H @ ye) # eq.(19)
    cg_tilde = cg + 0.5*np.trace(cuR + cdR - 2*cqL) # eq.(20)
    cW_tilde = cW - 0.5*np.trace(3*cqL + clL)
    cB_tilde = cB + np.trace(4/3*cuR + 1/3*cdR - 1/6*cqL + ceR - 1/2*clL)

    # eq(5) choosing betaQ = betaL = 0
    betau = -1
    betad = 1
    betae = 1
    CF = lambda N: 0.5*(N**2-1)/N

    # eq(18)
    beta_cqL = 1/(32*np.pi**2)*((yu @ yu.H + yd @ yd.H) @ cqL + cqL @ (yu @ yu.H + yd @ yd.H))-1/(16*np.pi**2)*(yu @ cuR @ yu.H + yd @ cdR @ yd.H) -3/(4*np.pi**2)*np.eye(3)*(alpha_s**2*CF(3)*cg_tilde + alpha_2**2*CF(2)*cW_tilde + alpha_1**2*(1/6)**2*cB_tilde)
    beta_cuR = 1/(16*np.pi**2)*(yu.H @ yu @ cuR + cuR @ yu.H @ yu)-1/(8*np.pi**2)*(yu.H @ cqL @ yu)+1/(4*np.pi**2)*np.eye(3)*(0.5*betau*X+3*alpha_s**2*CF(3)*cg_tilde + 3*alpha_1**2*(2/3)**2*cB_tilde)
    beta_cdR = 1/(16*np.pi**2)*(yd.H @ yd @ cdR + cdR @ yd.H @ yd)-1/(8*np.pi**2)*(yd.H @ cqL @ yd)+1/(4*np.pi**2)*np.eye(3)*(0.5*betad*X+3*alpha_s**2*CF(3)*cg_tilde + 3*alpha_1**2*(-1/3)**2*cB_tilde)
    beta_clL = 1/(32*np.pi**2)*(ye @ ye.H @ clL + clL @ ye @ ye.H)-1/(16*np.pi**2)*(ye @ ceR @ ye.H) -3/(4*np.pi**2)*np.eye(3)*(alpha_2**2*CF(2)*cW_tilde + alpha_1**2*(-1/2)**2*cB_tilde)
    beta_ceR = 1/(16*np.pi**2)*(ye.H @ ye @ ceR + ceR @ ye.H @ ye)-1/(8*np.pi**2)*(ye.H @ clL @ ye)+1/(4*np.pi**2)*np.eye(3)*(0.5*betae*X + 3*alpha_1**2*(-1)**2*cB_tilde)
    return dict2column({'cqL': beta_cqL, 'cuR': beta_cuR, 'cdR': beta_cdR, 'clL': beta_clL, 'ceR': beta_ceR})

def rge_leadinglog(couplings, scale_in, scale_out=mZ):
    return column2dict(dict2column(couplings) + beta(np.log(scale_in), dict2column(couplings))*np.log(scale_out/scale_in) )

def rge_ivp(couplings, scale_in, scale_out=mZ):
    sol = solve_ivp(beta, (np.log(scale_in), np.log(scale_out)), dict2column(couplings))
    return column2dict(sol.y[:,-1])

def matching(couplings, mu0=mZ):
    #TODO: matching contributions to k's
    xt = mt**2/mW**2
    ctt = couplings['cuR'][2,2] - couplings['cqL'][2,2] # eq(25)
    cZZ = s2w**2*couplings['cB'] + (1-s2w)**2*couplings['cW'] # eq(40)
    cgammaZ = (1-s2w)*couplings['cW']-s2w*couplings['cB'] # eq(40)

    delta1 = -11/3

    # eq(59)
    def Delta_kF(t3f, qf):
        res_yuk = 3*yuk_t**2/(8*np.pi**2)*ctt*(t3f-qf*s2w)*np.log(mu0**2/mt**2)
        res_WW = couplings['cW']/(2*s2w**2)*(np.log(mu0**2/mW**2)+0.5+delta1)
        res_gammaZ = 2*cgammaZ/(s2w*(1-s2w))*qf*(t3f-qf*s2w)*(np.log(mu0**2/mZ**2)+1.5+delta1)
        res_ZZ = cZZ/(s2w**2*(1-s2w)**2)*(t3f-qf*s2w)**2*(np.log(mu0**2/mZ**2)+0.5+delta1)
        return res_yuk + 3*alpha_em(mu0)/(8*np.pi**2)*(res_WW + res_gammaZ + res_ZZ)

    # eq(59)
    def Delta_kf(qf):
        res_yuk = -3*yuk_t**2/(8*np.pi**2)*ctt*qf*s2w*np.log(mu0**2/mt**2)
        res_gammaZ = 2*cgammaZ/(1-s2w)*qf**2*(np.log(mu0**2/mZ**2)+1.5+delta1)
        res_ZZ = -cZZ/((1-s2w)**2)*qf**2*(np.log(mu0**2/mZ**2)+0.5+delta1)
        return res_yuk + 3*alpha_em(mu0)/(8*np.pi**2)*(res_gammaZ + res_ZZ)
    
    # eq(60)
    DeltakD_lfv = (np.einsum('im,nj,kn->ijkm', Vckm.H, Vckm, couplings['cqL'])[:,:,2,2]+np.einsum('im,nj,mk->ijkn', Vckm.H, Vckm, couplings['cqL'])[:,:,2,2])*(-0.25*np.log(mu0**2/mt**2)-3/8+0.75*(1-xt+np.log(xt))/(1-xt)**2)
    DeltakD_lfv += couplings['cqL'][2,2]*np.einsum('ia,bj->ijab', Vckm.H, Vckm)[:,:,2,2]
    DeltakD_lfv += couplings['cuR'][2,2]*np.einsum('ia,bj->ijab', Vckm.H, Vckm)[:,:,2,2]
    DeltakD_lfv += -3*alpha_em(mu0)/(2*np.pi*s2w)*couplings['cW']*(1-xt+xt*np.log(xt))/((1-xt)**2)*np.einsum('ia,bj->ijab', Vckm.H, Vckm)[:,:,2,2]
    DeltakD_lfv *= (0.25*yuk_t/np.pi)**2

    return {'cg': couplings['cg'],
            'cgamma': couplings['cW']+couplings['cB'],
            'kD':Vckm.H @ couplings['cqL'] @ Vckm + Delta_kF(-0.5, -1/3)*np.eye(3) + DeltakD_lfv,
            'kU': couplings['cqL'][:2,:2] + Delta_kF(0.5, 2/3)*np.eye(2),
            'kE': couplings['clL'] + Delta_kF(-0.5, -1)*np.eye(3),
            'knu': couplings['clL'] + Delta_kF(0.5, 0)*np.eye(3),
            'ku': couplings['cuR'][:2,:2] + Delta_kf(2/3)*np.eye(2),
            'kd': couplings['cdR'] + Delta_kf(-1/3)*np.eye(3),
            'ke': couplings['ceR'] + Delta_kf(-1)*np.eye(3),
           }

def dict2column_ew(couplings):
    v = []
    for f in ['kD', 'kE', 'knu', 'kd', 'ke']:
        m = couplings[f]
        s = np.zeros(6, dtype=complex)
        s[0] = m[0,0]
        s[1] = m[0,1]
        s[2] = m[0,2]
        s[3] = m[1,1]
        s[4] = m[1,2]
        s[5] = m[2,2]
        v.append(s)
    for f in ['kU', 'ku']:
        m = couplings[f]
        s = np.zeros(3, dtype=complex)
        s[0] = m[0,0]
        s[1] = m[0,1]
        s[2] = m[1,1]
        v.append(s)
    v.append([couplings['cg'], couplings['cgamma']])
    return np.concatenate(v)

def column2dict_ew(v):
    d = {}
    for i, f in enumerate(['kD', 'kE', 'knu', 'kd', 'ke']):
        m = np.matrix(np.zeros((3,3), dtype=complex))
        m[0,0] = v[6*i]
        m[0,1] = v[6*i+1]
        m[1,0] = np.conjugate(v[6*i+1])
        m[0,2] = v[6*i+2]
        m[2,0] = np.conjugate(v[6*i+2])
        m[1,1] = v[6*i+3]
        m[1,2] = v[6*i+4]
        m[2,1] = np.conjugate(v[6*i+4])
        m[2,2] = v[6*i+5]
        d |= {f: m}
    for i, f in enumerate(['kU', 'ku']):
        m = np.matrix(np.zeros((2,2), dtype=complex))
        m[0,0] = v[30+3*i]
        m[0,1] = v[30+3*i+1]
        m[1,0] = np.conjugate(v[30+3*i+1])
        m[1,1] = v[30+3*i+2]
        d |= {f: m}
    d |= {'cg': v[36], 'cgamma': v[37]}
    return d

def beta_ew(mu0, vec):
    couplings = column2dict_ew(vec)
    kD = couplings['kD']
    kd = couplings['kd']
    kU = couplings['kU']
    ku = couplings['ku']
    kE = couplings['kE']
    ke = couplings['ke']

    wSM = wilson.classes.SMEFT(wilson.wcxf.WC('SMEFT', 'Warsaw', mu0, {})).C_in
    alpha_s = wSM['gs']**2/(4*np.pi)
    
    # eq(50)
    cdd = np.diag(kd - kD)
    cuu = np.diag(ku - kU)
    cee = np.diag(ke - kE)

    # eq(58)
    cg_tilde = couplings['cg']+0.5*np.sum(cdd)+0.5*np.sum(cuu)
    cgamma_tilde = couplings['cgamma']+np.sum(cee)+3*(2/3)**2*np.sum(cuu)+3*(-1/3)**2*np.sum(cdd)
    return dict2column_ew({'cg': 0, 'cgamma': 0, 'knu': np.zeros((3,3)),
           'ku': 3*alpha_s**2/np.pi**2*cg_tilde * np.eye(2) + 3*alpha_em(mu)/(4*np.pi**2)*(2/3)**2*cgamma_tilde * np.eye(2),
           'kU': -3*alpha_s**2/np.pi**2*cg_tilde * np.eye(2) - 3*alpha_em(mu)/(4*np.pi**2)*(2/3)**2*cgamma_tilde * np.eye(2),
           'kd': 3*alpha_s**2/np.pi**2*cg_tilde * np.eye(3) + 3*alpha_em(mu)/(4*np.pi**2)*(-1/3)**2*cgamma_tilde * np.eye(3),
           'kD': -3*alpha_s**2/np.pi**2*cg_tilde * np.eye(3) - 3*alpha_em(mu)/(4*np.pi**2)*(-1/3)**2*cgamma_tilde * np.eye(3),
           'ke': 3*alpha_em(mu)/(4*np.pi**2)*cgamma_tilde * np.eye(3),
           'kE': -3*alpha_em(mu)/(4*np.pi**2)*cgamma_tilde * np.eye(3),
           })

def rge_ew_leadinglog(couplings, scale_in, scale_out):
    return column2dict_ew(dict2column_ew(couplings) + beta_ew(np.log(scale_in), dict2column_ew(couplings))*np.log(scale_out/scale_in) )

def rge_ew_ivp(couplings, scale_in, scale_out):
    sol = solve_ivp(beta_ew, (np.log(scale_in), np.log(scale_out)), dict2column_ew(couplings))
    return column2dict_ew(sol.y[:,-1])

def match_run(couplings, scale_in, scale_out, leadinglog=False, ew_scale=mZ):
    if scale_in < scale_out or scale_in < 0 or scale_out < 0:
        raise ValueError("Wrong energy scales")
    if scale_in > ew_scale and scale_out > ew_scale:
        if leadinglog:
            return rge_leadinglog(couplings, scale_in, scale_out)
        return rge_ivp(couplings, scale_in, scale_out)
    if scale_in < ew_scale:
        if leadinglog:
            k = rge_leadinglog(couplings, scale_in, scale_out)
        else:
            k = rge_ivp(couplings, scale_in, scale_out)
    else:
        if leadinglog:
            coup_ew = rge_leadinglog(couplings, scale_in, ew_scale)
            coup_ewm = matching(coup_ew, ew_scale)
            k = rge_ew_leadinglog(coup_ewm, ew_scale, scale_out)
        else:
            coup_ew = rge_ivp(couplings, scale_in, ew_scale)
            coup_ewm = matching(coup_ew, ew_scale)
            k = rge_ew_ivp(coup_ewm, ew_scale, scale_out)
            
    return {'cg': k['cg'], 'cgamma': k['cgamma'],
            'cVu': k['ku'] + k['kU'],
            'cAu': k['ku'] - k['kU'],
            'cVd': k['kd'] + k['kD'],
            'cAd': k['kd'] - k['kD'],
            'cVe': k['ke'] + k['kE'],
            'cAe': k['ke'] - k['kE'],
            'cLnu': k['knu']
           }   