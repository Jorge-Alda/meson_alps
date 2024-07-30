"""Classes for RG evolution of the ALP couplings"""

import numpy as np


bases_above = ['derivative_above']
class ALPcouplings:
    """Container for ALP couplings.

    Members
    -------
    values : dict
        dict containing the ALP couplings.

    scale : float
        Energy scale where the couplings are defined, in GeV.

    basis : str
        Basis in which the couplings are defined. The available bases are:

        * 'derivative_above':
            Basis with the explicitly shift-symmetric couplings of the fermion currents to the derivative of the ALP; above the EW scale.

    Raises
    ------
    ValueError
        If attempting to translate to an unrecognized basis.
    """
    def __init__(self, values: dict, scale:float, basis:str):
        """Constructor method

        Parameters
        -------
        values : dict
            dict containing the ALP couplings.

        scale : float
            Energy scale where the couplings are defined, in GeV.

        basis : str
            Basis in which the couplings are defined. The available bases are:

            - 'derivative_above':
                Basis with the explicitly shift-symmetric couplings of the fermion currents to the derivative of the ALP; above the EW scale.

        Raises
        ------
        ValueError
            If attempting to translate to an unrecognized basis.

        TypeError
            If attempting to assign a non-numeric value
        """
        if basis == 'derivative_above':
            self.scale = scale
            self.basis = basis
            values = {'cg':0, 'cB': 0, 'cW':0, 'cqL': 0, 'cuR':0, 'cdR':0, 'clL':0, 'ceR':0} | values
            for c in ['cqL', 'cuR', 'cdR', 'clL', 'ceR']:
                if isinstance(values[c], (float, int)):
                    values[c] = np.matrix(values[c]*np.eye(3))
                elif isinstance(values[c], (np.ndarray, np.matrix, list)):
                    values[c] = np.matrix(values[c]).reshape([3,3])
                else:
                    raise TypeError
            for c in ['cg', 'cW', 'cB']:
                if not isinstance(values[c], (int, float)):
                     raise TypeError
            self.values = {c: values[c] for c in ['cg', 'cB', 'cW', 'cqL', 'cuR', 'cdR', 'clL', 'ceR']}
        else:
            raise ValueError('Unknown basis')
    
    def __add__(self, other: 'ALPcouplings') -> 'ALPcouplings':
        if self.basis == other.basis:
            return ALPcouplings({k: self.values[k]+other.values[k] for k in self.values.keys()}, min(self.scale, other.scale), self.basis)
        
    def __mul__(self, a: float) -> 'ALPcouplings':
            return ALPcouplings({k: a*self.values[k] for k in self.values.keys()}, self.scale, self.basis)
                                
    def __rmul__(self, a: float) -> 'ALPcouplings':
            return ALPcouplings({k: a*self.values[k] for k in self.values.keys()}, self.scale, self.basis)
    
    def __getitem__(self, name: str):
         return self.values[name]
    
    def __setitem__(self, name: str, val):
        if self.basis == 'derivative_above':
            if name in ['cg', 'cW', 'cB']:
                if isinstance(val, (float, int)):
                    self.values[name] = val
                else:
                    raise TypeError
            elif name in ['cqL', 'cuR', 'cdR', 'clL', 'ceR']:
                if isinstance(val, (float, int)):
                    self.values[name] = val * np.eye(3)
                elif isinstance(val, (np.ndarray, np.matrix, list)):
                    self.values[name] = np.matrix(val).reshape([3,3])
                else:
                    raise TypeError
            else:
                raise KeyError

    def translate(self, basis: str) -> 'ALPcouplings':
        """Translate the couplings to another basis at the same energy scale.
        
        Parameters
        ----------
        basis : str
            Target basis to translate.

        Returns
        -------
        a : ALPcouplings
            Translated couplings.

        Raises
        ------
        ValueError
            If attempting to translate to an unrecognized basis.
        """
        if basis == self.basis:
            return self
        else:
            raise ValueError('Unknown basis')
        
    def _toarray(self) -> np.ndarray:
        "Converts the object into a vector of coefficientes"
        if self.basis == 'derivative_above':
            return np.hstack([np.asarray(self.values[c]).ravel() for c in ['cqL', 'cuR', 'cdR', 'clL', 'ceR', 'cg', 'cB', 'cW']])
    
    @classmethod
    def _fromarray(cls, array: np.ndarray, scale: float, basis: str) -> 'ALPcouplings':
        if basis == 'derivative_above':
            vals = {}
            for i, c in enumerate(['cqL', 'cuR', 'cdR', 'clL', 'ceR']):
                vals |= {c: array[9*i:9*(i+1)].reshape([3,3])}
            vals |= {'cg': float(array[45]), "cB": float(array[46]), 'cW': float(array[47])}
            return ALPcouplings(vals, scale, basis)
        
    def match_run(self, scale_out: float, basis: str, integrator: str='scipy', beta: str='full', matching_scale: float = 100.0) -> 'ALPcouplings':
        from . import run_high 
        if scale_out > self.scale:
            raise ValueError("The final scale must be smaller than the initial scale.")
        if scale_out == self.scale:
            return self.translate(basis)
        if scale_out < matching_scale:
            raise NotImplemented
        if basis in bases_above and self.basis in bases_above:
            if beta == 'ytop':
                betafunc = run_high.beta_ytop
            elif beta == 'full':
                betafunc = run_high.beta_full
            else:
                raise KeyError(beta)
            if integrator == 'scipy':
                return run_high.run_scipy(self, betafunc, scale_out)
            if integrator == 'leadinglog':
                return run_high.run_leadinglog(self, betafunc, scale_out)
            else:
                raise KeyError(integrator)
        else:
            raise KeyError(basis)


