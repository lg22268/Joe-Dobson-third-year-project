#/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# *   GOBNILP (Python version) Copyright (C) 2019 James Cussens           *
# *                                                                       *
# *   This program is free software; you can redistribute it and/or       *
# *   modify it under the terms of the GNU General Public License as      *
# *   published by the Free Software Foundation; either version 3 of the  *
# *   License, or (at your option) any later version.                     *
# *                                                                       *
# *   This program is distributed in the hope that it will be useful,     *
# *   but WITHOUT ANY WARRANTY; without even the implied warranty of      *
# *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU    *
# *   General Public License for more details.                            *
# *                                                                       *
# *   You should have received a copy of the GNU General Public License   *
# *   along with this program; if not, see                                *
# *   <http://www.gnu.org/licenses>.                                      *
"""
    Code for computing local scores from continuous data
"""

__author__ = "James Cussens, Michael Bramhall"
__email__ = "james.cussens@york.ac.uk"



from math import log, pi
from scipy.special import gammaln
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.stats import norm

class ContinuousData:
    """
    Complete continuous data
    """

    def __init__(self, data, varnames=None):
        '''Continuous data

        Args:
            data (numpy.ndarray/str) : The data (either as an array or a filename containing the data)
            varnames (iterable/None) : The names of the variables. If not given
             (=None) and `data` is not a file having the variable names as a header 
             then the variables are named X1, X2, X3, etc
        '''
        if type(data) == str:
            skiprows = 0
            with open(data) as f:            
                for line in f:
                    if not line.startswith('#'):
                        varnames = line.strip().split()
                        skiprows += 1
                        break
                    else:
                        skiprows +=1
            data = np.loadtxt(data, dtype = float, comments = '#', skiprows = skiprows)
        else:
            if type(data) == pd.DataFrame:
                varnames = list(data.columns)
            data = np.array(data, dtype=float)
        self._data = data
        
        try:
            n, p = data.shape
        except ValueError:
            raise ValueError("Data must be a 2-d array")
            
        if varnames is None:
            varnames = tuple(('X{0}'.format(i+1) for i in range(p)))
        if len(varnames) != p:
            raise ValueError("Expected {0} variable names, got {1}".format(p,len(varnames)))
    
        # store everything
        self._n = n
        self._p = p
        self._variables = varnames
        self._varidx = {}
        for i, v in enumerate(varnames):
            self._varidx[v] = i
        self._cache = {}

    def variables(self):
        '''
        Returns:
         list : The variable names
        '''
        return self._variables

    def rawdata(self):
        '''
        The data without any information about variable names.

        Returns:
         numpy.ndarray: The data
        '''
        return self._data

    def data(self):
        '''
        The data as a Pandas dataframe.

        Returns:
         pandas.DataFrame: The data
        '''

        return pd.DataFrame(self._data,columns=self._variables)
    
    def varidx(self):
        '''
        Returns:
         dict : Maps a variable name to its position in the list of variable names.
        '''

        return self._varidx

class GaussianLL(ContinuousData):
    """
    Continuous data with attributes and methods for Gaussian log-likelihood scoring
    """

    def __init__(self, data):
        '''Create a GaussianLL scoring object

        Args:
            data (ContinuousData) : The data
        '''
        self.__dict__.update(data.__dict__)

    def score(self,child,parents):
        '''The Gaussian log-likelhood score for a given family, plus a dummy upper bound

        Args:
         child (str): The child variable
         parents (iter) : The parents
       
        Returns:
         tuple: First element of tuple is the Gaussian log-likelihood score for the family for current data
          Second element is None
        '''
        response_idx = self._varidx[child]
        y = self._data[:,response_idx]
        if len(parents) == 0:
            resids = np.mean(y) - y
        else:
            predictors_idx = np.array([self._varidx[v] for v in parents],dtype=np.uint32)
            X = self._data[:,predictors_idx]
            reg = LinearRegression().fit(X,y)
            preds = reg.predict(X)
            resids = y-preds
        sd = np.std(resids)
        return norm.logpdf(resids,scale=sd).sum(), None
        
    
class BGe(ContinuousData):
    """
    Continuous data with attributes and methods for BGe scoring
    """
    def __init__(self, data, nu=None, alpha_mu = 1.0, alpha_omega = None, prior_matrix=None):
        '''Create a BGe scoring object

        Args:
            data (ContinuousData) : The data
            nu (numpy.ndarray/None) : the mean vector for the normal part of the 
             normal-Wishart prior. If not given (=None), then the sample mean
             is used.
            alpha_mu (float) : imaginary sample size for the normal part of the
              normal-Wishart prior. 
            alpha_omega (int/None) : The degrees of freedom for the Wishart 
             part of the normal-Wishart prior. Must exceed p-1 where p is the number of 
             variables. If not given (=None) then `alpha_omega` is set to p+2.
            prior_matrix (numpy.ndarray/None) : The prior matrix 'T'
             for the Wishart part of the normal-Wishart prior. If not given (=None), then
             this is set to t*I_n where t = alpha_mu*(alpha_omega-n-1)/(alpha_mu+1)
        '''
        
        self.__dict__.update(data.__dict__)
        p = self._p
        n = self._n
        data = self._data
        
        # No need to explicitly represent nu if it the sample mean vector
        if nu is not None and len(nu) != p:
            raise ValueError("nu is wrong length. Expected length is {0}, but got length of {1}".format(p,len(nu)))

        if not alpha_mu > 0:
                raise ValueError("alpha_mu must be positive, but is {0}".format(alpha_mu))
        
        if alpha_omega is None:
            alpha_omega = self._p + 2
        else:
            if type(alpha_omega) != int:
                raise ValueError("alpha_omega must be an integer but is {1}".format(alpha_omega))
            if not alpha_omega > p - 1:
                raise ValueError("alpha_omega must exceed p-1 = {0}, but is {1}".format(p-1,alpha_omega))

        if prior_matrix is None:
            explicit_prior_matrix = np.zeros((p,p))
            t = (alpha_mu * (alpha_omega - p - 1.0)) / (alpha_mu + 1.0)
            np.fill_diagonal(explicit_prior_matrix, t)
        else:
            explicit_prior_matrix = prior_matrix
            d1, d2 = explicit_prior_matrix.shape
            if d1 != p or d2 != p:
                raise ValueError("prior_matrix must be {0} X {0} but is {1} X {2}".format(p,d1,d2))
            # need to add check for prior_matrix being positive definite

        # for debugging ...
        #print(explicit_prior_matrix)
            
        # compute (log) prefactors for each possible size of parent set
        log_prefactors = []
        const_log_prefactor = 0.5 * (log(alpha_mu) - log(n + alpha_mu))
        for pasize in range(p):
            log_prefactor = (
                const_log_prefactor
                + gammaln(0.5*(n + alpha_omega - p + pasize + 1))
                - gammaln(0.5*(alpha_omega - p + pasize + 1))
                - (0.5*n) * log(pi))
            log_prefactors.append(log_prefactor)
        # If using 'default' prior matrix then include ratio of dets of prior matrices here:
        if prior_matrix is None:
            logt = log(t)
            const_term = 0.5 * (alpha_omega - p + 1) 
            for pasize in range(p):
                log_prefactors[pasize] += (const_term + pasize) * logt

        # compute posterior matrix 'R'
        # need rowvar=F since each row (not column) is a datapoint
        s_n = (n-1) * np.cov(data,rowvar=False)
        posterior_matrix = np.add(explicit_prior_matrix,s_n)
        if nu is not None:
            diff_vec = np.subtract(nu,np.mean(data,axis=0))
            # (4) in Kuipers et al is wrong, since it has alpha_omega
            # when it should be alpha_mu in (n*alpha_mu)/(n+alpha_mu)
            posterior_matrix = np.add(posterior_matrix,
                                      ((n*alpha_mu)/(n+alpha_mu))*np.outer(diff_vec,diff_vec))
        # for debugging ...
        #print(posterior_matrix)
            
        # store everything
        self._alpha_mu = alpha_mu
        self._alpha_omega = alpha_omega
        self._log_prefactors = log_prefactors
        self._prior_matrix = prior_matrix # NB could be None
        self._posterior_matrix = posterior_matrix
        self._cache = {}

    
    def bge_component(self,vs):
        '''Compute the BGe component for given variables

        The BGe score for a family child<-parents is the component for child+parents
        minus the component for parents (+ a constant term which just depends on the number
        of parents).

        Args:
         vs (iter): Variable names
        
        Returns:
         float: The BGe component for the given variables
        '''
        vset = frozenset(vs)
        try:
            component = self._cache[vset]
        except KeyError:
            # since we ultimately want absolute value of determinant, no need to worry about order
            # unless, perhaps, there a performance issue?
            indices = [self._varidx[v] for v in vs]
            # Get 'R_PP' or 'R_QQ'
            array_indices = np.ix_(indices,indices)
            posterior_matrix = self._posterior_matrix[array_indices]
            component = -(0.5 *
                         (self._n + self._alpha_omega - self._p + len(vs)) *
                         (np.linalg.slogdet(posterior_matrix)[1]))
            if self._prior_matrix is not None:
                # Get 'T_PP' or 'T_QQ'
                prior_matrix = self._prior_matrix[array_indices]
                component += (0.5 *
                              (self._alpha_omega - self._p + len(vs)) *
                              (np.linalg.slogdet(prior_matrix)[1]))
            self._cache[vset] = component
        return component
            
    def bge_score(self,child,parents):
        '''The BGe score for a given family, plus upper bound

        Args:
         child (str): The child variable
         parents (iter) : The parents
       
        Returns:
         tuple: First element of tuple isf the BGe score for the family for current data (using current hyperparameters)
          Second element is an upper bound.
        '''
        return (self._log_prefactors[len(parents)] +
                self.bge_component(list(parents)+[child]) -
                self.bge_component(parents), None)

