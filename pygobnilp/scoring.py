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
    Code for computing local scores from data
"""

__author__ = "Josh Neil, James Cussens"
__email__ = "james.cussens@bristol.ac.uk"

from math import lgamma, log, pi
from itertools import combinations

from scipy.special import digamma, gammaln
from scipy.stats import norm, entropy
from scipy.optimize import linprog

from sklearn.linear_model import LogisticRegression

import numpy as np
import pandas as pd

from numba import jit, njit

import tempfile
import os

try:
    import adtree
    adtree_available = True
except ImportError as e:
    adtree_available = False
    print("C ADTree implementation unavailable.")
    print(e)

# START functions for contabs

@jit(nopython=True)
def marginalise_uniques_contab(unique_insts, counts, cols):
    '''
    Marginalise a contingency table represented by unique insts and counts
    '''
    marg_uniqs, indices = np.unique(unique_insts[:,cols], return_inverse=True)
    marg_counts = np.zeros(len(marg_uniqs),dtype=np.uint32)
    for i in range(len(counts)):
        marg_counts[indices[i]] += counts[i]
    return marg_uniqs, marg_counts

@jit(nopython=True)
def make_contab(data, counts, cols, arities, maxsize):
    '''
    Compute a marginal contingency table from data or report
    that the desired contingency table would be too big.

    All inputs except the last are arrays of unsigned integers

    Args:
     data (numpy array): the unique datapoints as a 2-d array, each row is a datapoint, assumed unique
     counts (numpy array): the count of how often each unique datapoint occurs in the original data
     cols (numpy array): the columns (=variables) for the marginal contingency table.
      columns must be ordered low to high
     arities (numpy array): the arities of the variables (=columns) for the contingency table
      order must match that of `cols`.
     maxsize (int): the maximum size (number of cells) allowed for a contingency table

    Returns:
     tuple: 1st element is of type ndarray: 
      If the contingency table would have more elements than `maxsize' then the array is empty
      (and the 2nd element of the tuple should be ignored)
      else an array of counts of length equal to the product of the `arities`.
      Counts are in lexicographic order of the joint instantiations of the columns (=variables)
      2nd element: the 'strides' for each column (=variable)
    '''
    p = len(cols)
    #if arities = (2,3,3) then strides = 9,3,1
    #if row is (2,1,2) then index is 2*9 + 1*3 + 2*1
    strides = np.empty(p,dtype=np.uint32)
    idx = p-1
    stride = 1
    while idx > -1:
        strides[idx] = stride
        stride *= arities[idx]
        if stride > maxsize:
            return np.empty(0,dtype=np.uint32), strides
        idx -= 1
    contab = np.zeros(stride,dtype=np.uint32)
    for rowidx in range(data.shape[0]):
        idx = 0
        for i in range(p):
            idx += data[rowidx,cols[i]]*strides[i]
        contab[idx] += counts[rowidx]
    return contab, strides

@jit(nopython=True)
def _compute_ll_from_flat_contab(contab,strides,child_idx,child_arity):
    child_stride = strides[child_idx]
    ll = 0.0
    child_counts = np.empty(child_arity,dtype=np.int32)
    contab_size = len(contab)
    for i in range(0,contab_size,child_arity*child_stride):
        for j in range(i,i+child_stride):
            n = 0
            for k in range(child_arity):
                count = contab[j]
                child_counts[k] = count
                n += count
                j += child_stride
            if n > 0:
                for c in child_counts:
                    if c > 0:
                        ll += c * log(c/n)
    return ll

@jit(nopython=True)
def _compute_ll_from_unique_contab(data,counts,n_uniqs,pa_idxs,child_arity,orig_child_col):
    child_counts = np.zeros((n_uniqs,np.int64(child_arity)),dtype=np.uint32)
    for i in range(len(data)):
        child_counts[pa_idxs[i],data[i,orig_child_col]] += counts[i]
    ll = 0.0
    for i in range(len(child_counts)):
        #n = child_counts[i,:].sum() #to consider
        n = 0
        for k in range(child_arity):
            n += child_counts[i,k]
        for k in range(child_arity):
            c = child_counts[i,k]
            if c > 0:
                ll += c * log(c/n)
    return ll

@jit(nopython=True)
def compute_bdeu_component(data, counts, cols, arities, alpha, maxflatcontabsize):
    contab = make_contab(data, counts, cols, arities, maxflatcontabsize)[0]
    if len(contab) > 0:
        alpha_div_arities = alpha / len(contab)
        non_zero_count = 0
        score = 0.0
        for count in contab:
            if count != 0:
                non_zero_count += 1
                score -= lgamma(alpha_div_arities+count) 
        score += non_zero_count*lgamma(alpha_div_arities)  
        return score, non_zero_count


# START functions for upper bounds

@jit(nopython=True)
def lg(n,x):
    return lgamma(n+x) - lgamma(x)

def h(counts):
    '''
    log P(counts|theta) where theta are MLE estimates computed from counts
    ''' 
    tot = sum(counts)
    #print(tot)
    res = 0.0
    for n in counts:
        if n > 0:
            res += n*log(n/tot)
    return res

def chisq(counts):
    tot = sum(counts)
    t = len(counts)
    mean = tot/t #Python 3 - this creates a float
    chisq = 0.0
    for n in counts:
        chisq += (n - mean)**2
    return chisq/mean

def hsum(distss):
    res = 0.0
    for dists in distss:
        res += sum([h(d) for d in dists])
    return res

@jit(nopython=True)
def fa(dist,sumdist,alpha,r):
    #res = -lg(sumdist,alpha)
    res = lgamma(alpha) - lgamma(sumdist+alpha)
    alphar = alpha/r
    k = 0
    for n in dist:
        if n > 0:
            #res += lg(n,alphar)
            #res += (lgamma(n+alphar) - lgamma(alphar))
            res += lgamma(n+alphar)
            k += 1
    return res - k*lgamma(alphar)

def diffa(dist,alpha,r):
    """Compute the derivative of local-local BDeu score

    numba does not support the scipy.special functions -- we are using the
    digamma function in defining the entropy of the Chisq
    distribution. For this end I added a python script which contains code
    for a @vectorize-d digamma function. If we need to use anything from
    scipy.special we will have to write it up ourselves.

    """
    args = [n+alpha/r for n in dist] + [alpha,sum(dist)+alpha,alpha/r]
    z = digamma(args)
    return sum(z[:r+1]) - z[r+1] - r*z[r+2] 

def onepositive(dist):
    """
    Is there only one positive count in `dist`?
    """
    npos = 0
    for n in dist:
        if n > 0:
            npos += 1
            if npos > 1:
                return False
    return True

@njit
def array_equal(a,b,pasize):
    for i in range(pasize):
        if a[i] != b[i]:
            return False
    return True

#@njit
#def get_elems(a,idxs):
#    return np.a

@njit
def upper_bound_james_fun(atoms_ints,atoms_floats,pasize,alpha,r,idxs):
    ub = 0.0
    local_ub = 0.0
    best_diff = 0.0
    pasize_r = pasize+r
    lr = -log(r)
    oldrow = atoms_ints[idxs[0]]
    for i in idxs:
        row = atoms_ints[i]
        
        if not array_equal(oldrow,row,pasize):
            ub += min(local_ub+best_diff,lr)
            best_diff = 0.0
            local_ub = 0.0
            oldrow = row

        mle = atoms_floats[i]
        local_ub += mle
        if row[-1]: # if chi-sq condition met
            diff = fa(row[pasize:pasize_r],row[pasize_r],alpha/2.0,r) - mle
            if diff < best_diff:
                best_diff = diff
    ub += min(local_ub+best_diff,lr)
    return ub


#@jit(nopython=True)
def ub(dists,alpha,r):
    '''
    Args:
        dists (iter): list of (child-value-counts,chisqtest,mles) lists, for some
          particular instantiation ('inst1') of the current parents.
          There is a child-value-counts list for each non-zero instantiation ('inst2') of the biggest
          possible superset of the current set of parents. inst1 and each inst2 have the same
          values for the current parents.
        alpha (float): ESS/(num of current parent insts)
        r (int): arity of the child

    Returns:
       float: An upper bound on the BDeu local score for (implicit) child
        over all possible proper supersets of the current parents
    '''
    naives = 0.0
    best_diff = 0.0
    for (dist,sumdist,ok_first,naive_ub) in dists:
        naives += naive_ub
        #iffirst_ub = naive_ub
        #iffirst_ub = min(len(dist)*-log(r),naive_ub)
        if ok_first:
            diff = fa(dist,sumdist,alpha/2.0,r) - naive_ub
            #iffirst_ub = min(iffirst_ub,fa(dist,sumdist,alpha/2.0,r))
            #diff = iffirst_ub - naive_ub
            if diff < best_diff:
                best_diff = diff
    return best_diff + naives


# @jit(nopython=True)
# def _ll_score(contab_generator, arities, variables, child, process_contab_function):
#     """
#         Parameters:

#         - `contab_generator`: Function used to generate the contingency tables for `variables`
#         - `arities`: A NumPy array containing the arity of every variable
#         - `variables`: A Numpy array of ints which are column indices for the family = (child + parents)
#         - `child`: The column index for the child 
#         - `process_contab_function`: Function called to do main computation

#         - Returns the fitted log-likelihood (LL) score 
#     """
#     contab = contab_generator.make_contab(variables)
#     score, non_zero_count = process_contab_function(arities, contab, variables, alpha_div_arities)             
    
#     return score, non_zero_count



#@jit(nopython=True)
def get_atoms(data,arities):
    '''
    Args: 
        data(np.array): Discrete data as a 2d array of ints

    Returns:
        list: a list `atoms`, where `atoms[i]` is a dictionary mapping instantations
         of variables other than i to a tuple with 3 elements:

          1. the child counts (n1, n2, .., nr) for that instantations
          2. n * the entropy for the empirical distribution (n1/n, n2/n, .., nr/n)
          3. Whether the chi-squared statistic for (n1, n2, .., nr) exceeds r-1 
        
        Chi-squared test comes from "Compound Multinomial Likelihood Functions are Unimodal: 
        Proof of a Conjecture of I. J. Good". Author(s): Bruce Levin and James Reeds
        Source: The Annals of Statistics, Vol. 5, No. 1 (Jan., 1977), pp. 79-87
    
        At least two of the ni must be positive for the inst-tuple pair to be included
        in the dictionary since only in that case is the inst-tuple useful for getting
        a good upper bound.
       
    '''
    fullinsts = []
    for i in range(data.shape[1]):
        fullinsts.append({})
    for row in data:
        row = tuple(row)
        for i, val in enumerate(row):
            fullinsts[i].setdefault(row[:i]+(0,)+row[i+1:],  # add dummy value for ith val
                                    [0]*arities[i])[val] += 1

    # now, for each child i, delete full insts which are 'deterministic'
    # i.e. where only one child value is non-zero
    newdkts_ints = []
    newdkts_floats = []
    for i, dkt in enumerate(fullinsts):
        #print('old',len(dkt))
        #newdkt = {}
        newdkt_ints = []
        newdkt_floats = []
        th = arities[i] - 1
        for inst, childcounts in dkt.items():
            if len([v for v in childcounts if v > 0]) > 1:
                newdkt_ints.append(inst+tuple(childcounts)+(sum(childcounts),chisq(childcounts) > th))
                #newdkt[inst] = (
                #    np.array(childcounts,np.uint64),sum(childcounts),
                #    chisq(childcounts) > th,
                #    h(childcounts))
                newdkt_floats.append(h(childcounts))
        newdkts_ints.append(np.array(newdkt_ints,np.uint64))
        newdkts_floats.append(np.array(newdkt_floats,np.float64))
        #print('new',len(newdkt))
    #fullinsts = newdkts

    #print(newdkts_ints[0])
    #sys.exit()

    #for x in newdkts_ints:
    #    print(x.shape)
    
    return newdkts_ints, newdkts_floats
    

def save_local_scores(local_scores, filename):
    variables = local_scores.keys()
    with open(filename, "w") as scores_file:
        scores_file.write(str(len(variables)))
        for child, dkt in local_scores.items():
            scores_file.write("\n" + child + " " + str(len(dkt.keys())))
            for parents, score in dkt.items():
                #highest_sup = None
                scores_file.write("\n" + str(score) + " " + str(len(parents)) +" "+ " ".join(parents))

def prune_local_score(this_score, parent_set, child_dkt):
    for other_parent_set, other_parent_set_score in child_dkt.items():
        if other_parent_set_score >= this_score and other_parent_set < parent_set:
            return True
    return False

def fromdataframe(df):
    cols = []
    arities = []
    varnames = []
    for varname, vals in df.items():
        varnames.append(varname)
        cols.append(vals.cat.codes)
        arities.append(len(vals.cat.categories))
    return np.transpose(np.array(cols,dtype=np.uint32)), arities, varnames

class Data:
    """
    Complete data (either discrete or continuous)

    This is an abstract class
    """

    def rawdata(self):
        '''
        The data without any information about variable names.

        Returns:
         numpy.ndarray: The data
        '''
        return self._data

    def variables(self):
        '''
        Returns:
         list : The variable names
        '''
        return self._variables

    def varidx(self):
        '''
        Returns:
         dict : Maps a variable name to its position in the list of variable names.
        '''
        return self._varidx

    
class DiscreteData(Data):
    """
    Complete discrete data
    """

    _value_type = np.uint8
    _arity_type = np.uint8
    _count_type = np.uint32
    
    def __init__(self, data_source, varnames = None, arities = None, binary = False):
        '''Initialises a `DiscreteData` object.

        If  `data_source` is a filename then it is assumed that:

            #. All values are separated by whitespace
            #. Empty lines are ignored
            #. Comment lines start with a '#'
            #. The first line is a header line stating the names of the 
               variables
            #. The second line states the arities of the variables
            #. All other lines contain the actual data

        Args:
          data_source (str/array_like/Pandas.DataFrame) : 
            Either a filename containing the data or an array_like object or
            Pandas data frame containing it.

          varnames (iter) : 
           Variable names corresponding to columns in the data.
           Ignored if `data_source` is a filename or Pandas DataFrame (since they 
           will supply the variable names). Otherwise if not supplied (`=None`)
           then variables names will be: X1, X2, ...

          arities (iter) : 
           Arities for the variables corresponding to columns in the data.
           Ignored if `data_source` is a filename or Pandas DataFrame (since they 
           will supply the arities). Otherwise if not supplied (`=None`)
           the arity for each variable will be set to the number of distinct values
           observed for that variable in the data.
        '''

        if type(data_source) == str:
            # If file is empty
            with open(data_source, "r") as file:    
                line = file.readline()
                while len(line) == 0 or len(line.rstrip()) == 0 or line[0] == '#':
                    if len(line) == 0:
                        self._data = []
                        self._variables = []
                        self._data_length = 0
                        return
                    line = file.readline()
                varnames = line.rstrip().split()
                line = file.readline().rstrip()
                while len(line) == 0 or line[0] == '#':
                    line = file.readline().rstrip()
                arities = np.array([int(x) for x in line.split()],dtype=self._arity_type)

                for arity in arities:
                    if arity < 2:
                        raise ValueError("This line: '{0}' is interpreted as giving variable arities but the value {1} is less than 2.".format(line,arity))

                # class whose instances are callable functions 'with memory'
                class Convert:
                    def __init__(self):
                        self._last = 0 
                        self._dkt = {}

                    def __call__(self,s):
                        if binary:
                            return s
                        try:
                            return self._dkt[s]
                        except KeyError:
                            self._dkt[s] = self._last
                            self._last += 1
                            return self._dkt[s]


                converter_dkt = {}
                for i in range(len(varnames)):
                    # trick to create a function 'with memory'
                    converter_dkt[i] = Convert()
                data = np.loadtxt(file,
                                  dtype=self._value_type,
                                  converters=converter_dkt,
                                  comments='#')

        elif type(data_source) == pd.DataFrame:
            data, arities, varnames = fromdataframe(data_source)
        else:
            data = np.array(data_source,dtype=self._value_type)
        self._data = data
        if arities is None:
            self._arities = np.array([x+1 for x in data.max(axis=0)],dtype=self._arity_type)
        else:
            self._arities = np.array(arities,dtype=self._arity_type)

        # ensure _variables is immutable _varidx is always correct.
        if varnames is None:
            self._variables = tuple(['X{0}'.format(i) for i in range(1,len(self._arities)+1)])
        else:
            # order of varnames determined by header line in file, if file used
            self._variables = tuple(varnames)

        self._unique_data, counts = np.unique(self._data, axis=0, return_counts=True)
        self._unique_data_counts = np.array(counts,self._count_type)
            
        self._maxflatcontabsize = 1000000
        self._contab = np.empty(self._maxflatcontabsize,dtype=np.int32)

        self._varidx = {}
        for i, v in enumerate(self._variables):
            self._varidx[v] = i
        self._data_length = data.shape[0]

        # create AD tree, if possible
        if adtree_available:
            self._adtree = adtree.adtree(10,1000,1000,
                                         np.array(self._data.flatten('F'),dtype=np.int32),
                                         np.array(self._arities,dtype=np.int32),
                                         self._data_length)
        
    def data(self):
        '''
        The data with all values converted to unsigned integers.

        Returns:
         pandas.DataFrame: The data
        '''

        df = pd.DataFrame(self._data,columns=self._variables)
        arities = self._arities
        for i, (name, data) in enumerate(df.items()):
            # ensure correct categories are recorded even if not
            # all observed in data
            df[name] = pd.Categorical(data,categories=range(arities[i]))
        return df
    
    def data_length(self):
        '''
        Returns:
         int: The number of datapoints in the data
        '''
        return self._data_length

    def arities(self):
        '''
        Returns:
         numpy.ndarray: The arities of the variables.
        '''
        return self._arities

    def arity(self,v):
        '''
        Args:
         v (str) : A variable name
        Returns:
         int : The arity of `v`
        '''

        return self._arities[self._varidx[v]]


    def contab(self,variables):
        cols = np.array([self._varidx[v] for v in variables], dtype=np.uint32)
        cols.sort() 
        return make_contab(self._unique_data,self._unique_data_counts,cols,self._arities[cols],self._maxflatcontabsize)[0]

    def make_contab_adtree(self,variables):
        '''
        Compute a marginal contingency table from data or report
        that the desired contingency table would be too big.
        
        Args:
         variables (iter): The variables in the marginal contingency table.

        Returns:
         tuple: 1st element is of type ndarray the first SIZE elements of which contain the required contingency table 
          2nd element: the value SIZE (product of arities of the variables), or -1 iff table too big
        '''
        cols = np.array([self._varidx[v] for v in variables], dtype=np.int32)
        cols.sort()
        size = adtree.contab(self._adtree,cols,self._contab)
        #print(flatcontab,flush=True)
        return self._contab, size
    
class ContinuousData(Data):
    """
    Complete continuous data
    """

    def __init__(self, data, varnames=None, header=True, comments='#', delimiter=None, standardise=False):
        '''Continuous data

        Args:
            data (numpy.ndarray/str) : The data (either as an array or a filename containing the data)
            varnames (iterable/None) : The names of the variables. If not given
             (=None) then if  `data` is a file having the variable names as a header then these are used
             else the variables are named X1, X2, X3, etc
            header (bool) : Ignored if `data` is not a filename. 
             Whether a header containing variable names is the first non-comment line in the file.
            comments (str) : Ignored if `data` is not a filename. Lines starting with this string are treated as comments.
            delimiter (None/str) : Ignored if `data` is not a filename. String used to separate values. If None then whitespace is used. 
            standardise (bool) : Whether to standardise the date to have mean 0 and sd = 1.
        '''
        if type(data) == str:
            skiprows = 0
            if header:
                with open(data) as f:            
                    for line in f:
                        if line.startswith(comments):
                             skiprows +=1
                        else:
                            vs = line.strip().split(sep=delimiter)
                            if varnames is None:  # only overwrite varnames if none given 
                                varnames = vs
                            skiprows += 1
                            break
            data = np.loadtxt(data, dtype = float, comments = comments, delimiter=delimiter, skiprows = skiprows)
        else:
            if type(data) == pd.DataFrame:
                varnames = list(data.columns)
            data = np.array(data, dtype=float)

        try:
            if len(data.shape) == 1:
                self._data = data
                self._variables = []
                self._data_length = 0
                return
            n, p = data.shape
        except ValueError:
            raise ValueError("Data must be a 2-d array")

        if standardise:
            self._data = (data - np.mean(data,axis=0)) / np.std(data,axis=0)
        else:
            self._data = data

        
        if varnames is None:
            varnames = ['X{0}'.format(i+1) for i in range(p)]
        if len(varnames) != p:
            raise ValueError("Expected {0} variable names, got {1}".format(p,len(varnames)))
    
        # store everything
        self._n = n
        self._p = p
        self._variables = tuple(varnames)
        self._cache = {}
        
        self._varidx = {}
        for i, v in enumerate(varnames):
            self._varidx[v] = i
        self._data_length = data.shape[0]


    def data(self):
        '''
        The data as a Pandas dataframe.

        Returns:
         pandas.DataFrame: The data
        '''

        return pd.DataFrame(self._data,columns=self._variables)
    
class MixedData(Data):
    

    def __init__(self, data, varnames=None, header=True, comments='#', delimiter=None, standardise=False, arities=None):
        
        self.discrete_cols = []
        self.continuous_cols = []
        self.arities = []
        self._is_discrete = []
        
        # Check arity line
        if type(data) == str:
            with open(data, "r") as file:
                line = file.readline().rstrip()
                while len(line) == 0 or line[0] == '#':
                    line = file.readline().rstrip()
                varnames = line.split()
                line = file.readline().rstrip()
                while len(line) == 0 or line[0] == '#':
                    line = file.readline().rstrip()
                arity_fields = line.split()
                for arity_index in range(len(arity_fields)):
                    if not arity_fields[arity_index].isdigit():
                        self.continuous_cols.append(arity_index)
                        self._is_discrete.append(False)      
                    else:
                        self.discrete_cols.append(arity_index)
                        self._is_discrete.append(True)
                        self.arities.append(int(arity_fields[arity_index]))
                # Create temp files for discrete and continuous 
                continuous_file = tempfile.NamedTemporaryFile(delete=False, mode='w+t')
                discrete_file = tempfile.NamedTemporaryFile(delete=False, mode='w+t')
                # Write header to respective files
                continuous_file.write(" ".join([varnames[i] for i in self.continuous_cols]) + '\n')
                discrete_file.write(" ".join([varnames[i] for i in self.discrete_cols]) + '\n')
                # Write arity to discrete file
                discrete_file.write(" ".join([arity_fields[i] for i in self.discrete_cols]) + '\n')
                # Write data to respective files
                for line in file:
                    fields = line.split()
                    continuous_file.write(" ".join([fields[i] for i in self.continuous_cols]) + '\n')
                    discrete_file.write(" ".join([fields[i] for i in self.discrete_cols]) + '\n')
                
                # Read line from file and split into discrete and continuous rows

                continuous_file_name = continuous_file.name
                discrete_file_name = discrete_file.name
                continuous_file.close()
                discrete_file.close()
                
                         
                # Construct discrete and continuous data objects
                self.continuous_data = ContinuousData(continuous_file_name)
                self.discrete_data = DiscreteData(discrete_file_name, binary=True)
                os.remove(continuous_file_name)
                os.remove(discrete_file_name)
                
                self._variables = tuple(varnames)
                
                # self._data = np.zeros((self.discrete_data.data_length(), len(self._variables)))
                # raw = self.discrete_data.rawdata().astype(np.float32)
                # normalized = raw / (np.array(self.arities) - 1)  # auto-broadcast
                # self._data[:, self.discrete_cols] = normalized
                # self._data[:, self.continuous_cols] = self.continuous_data.rawdata()
                
                self._data = np.zeros((max(self.discrete_data.data_length(),self.continuous_data._data_length), len(self._variables)))
                # raw = self.discrete_data.rawdata().astype(np.float32)
                # normalized = raw / (np.array(self.arities) - 1)  # auto-broadcast
                self._data[:, self.discrete_cols] = self.discrete_data.rawdata()
                self._data[:, self.continuous_cols] = self.continuous_data.rawdata()
                
                        
        elif type(data) == MixedData:
            self.continuous_data = data.continuous_data
            self.discrete_data = data.discrete_data
            self._variables = data._variables
            self._data = data._data
            self._is_discrete = data._is_discrete
            self.discrete_cols = data.discrete_cols
            self.continuous_cols = data.continuous_cols
            self.arities = arities
        
        else:
            print("Data type not supported")
            
        self.initialize_biases_from_data()
        self._varidx = {}
        for i, v in enumerate(self._variables):
            self._varidx[v] = i
            
    def initialize_biases_from_data(self):
        """
        Initializes c_i (biases for discrete variables) using empirical log-odds.
        """
        
        n_nodes = len(self._variables)
        self.biases = [0.0 for _ in range(n_nodes)]

        # for node in range(n_nodes):
        #     if self._is_discrete[node]:
        #         # Get all values for this node
        #         values = self._data[node]
        #         p_hat = np.mean(values)  # empirical P(x=1)

        #         # Clip to avoid log(0)#
        #         # TODO fix: divide by the arities to get a value between 0 and 1
        #         p_hat = np.clip(p_hat, 1e-5, 1 - 1e-5)
        #         c_i = np.log(p_hat / (1 - p_hat))
        #         self.biases[node] = c_i
        #         print(f"Node {node} bias: {c_i} p_hat: {p_hat}")
        #     else:
        #         self.biases[node] = 0.0  # or leave as 0 for continuous nodes

        print("Biases initialized from data: ", self.biases)
       
# START Classes for penalised log-likelihood 
    
class _AbsLLPenalised:
    '''Abstract class for discrete penalised log likelihood scores
    '''

    def __init__(self,data):
        '''Initialises a `_AbsLLPenalised` object.

        Args:
         data (DiscreteData/Continuous): data
        '''
        self.__dict__.update(data.__dict__)
        self._maxllh = {}
        if type(data) == ContinuousData:
            # compute and store the sample covariance matrix (the version that gives the MLE)
            self._cov = np.cov(self._data,rowvar=False,bias=True)
            self._gaussianll_cache = {}
            self._log2pi1 = log(2*pi) + 1
        if type(data) == DiscreteData:
            self._entropy_cache = {}

        for i, v in enumerate(self._variables):
            self._maxllh[v] = self.ll_score(v,self._variables[:i]+self._variables[i+1:])[0]

    
class AbsDiscreteLLScore(DiscreteData):
    '''Abstract class for discrete log likelihood scores
    '''

    def score(self,child,parents):
        '''
        Return LL score minus complexity penalty for `child` having `parents`, 
        and also upper bound on the score for proper supersets.

        To compute the penalty the number of joint instantations is multiplied by the arity
        of the child minus one. This value is then multiplied by log(N)/2 for BIC and 1 for AIC.

        Args:
         child (str): The child variable
         parents (iter): The parent variables

        Raises:
         ValueError: If the number of joint instantations of the parents would cause an overflow
          when computing the penalty
        
        Returns:
         tuple: The local score for the given family and an upper bound on the local score 
         for proper supersets of `parents`
        '''
        this_ll_score, numinsts = self.ll_score(child,parents)
        if numinsts is None:
            raise ValueError('Too many joint instantiations of parents {0} to compute penalty'.format(parents))
        penalty = numinsts * self._child_penalties[child]
        # number of parent insts will at least double if any added
        return this_ll_score - penalty, self._maxllh[child] - (penalty*2)

    def entropy(self,variables):
        '''
        Compute the entropy for the empirical distribution of some variables

        Args:
         variables (iter): Variables

        Returns:
         The entropy for the empirical distribution of `variables` and the number of joint
         instantiations of `variables` if not too big else None
        '''
        vset = frozenset(variables)
        try:
            return self._entropy_cache[vset]
        except KeyError:
            if adtree_available: # using global variable makes testing easier
                contab, size = self.make_contab_adtree(variables)
                if size > 0:
                    h = entropy(contab[:size])
                    self._entropy_cache[vset] = h, size
                    return h, size
                else:
                    # indicate too big
                    cols = np.array(sorted([self._varidx[x] for x in variables]), dtype=np.uint32)
                    contab = ()
            else:
                cols = np.array(sorted([self._varidx[x] for x in variables]), dtype=np.uint32)
                contab, strides = make_contab(self._unique_data, self._unique_data_counts, cols,
                                          self._arities[cols], self._maxflatcontabsize)
            # do it anyway!
            #contab, strides = make_contab(self._unique_data, self._unique_data_counts, cols,
            #                              self._arities[cols], self._maxflatcontabsize)
            #print(contab)
            
            numinsts = len(contab)
            if numinsts == 0:
                numinsts = None
                # need to resort to slower method, will move this to numba at some point
                uniqs, uniq_idxs = np.unique(self._unique_data[:,cols],axis=0,return_inverse=True)
                contab = np.zeros(len(uniqs),dtype=self._count_type)
                unique_data_counts = self._unique_data_counts
                for i in range(len(self._unique_data)):
                    contab[uniq_idxs[i]] += unique_data_counts[i]

            h = entropy(contab)
            self._entropy_cache[vset] = h, numinsts
            return h, numinsts

        
    def ll_score(self,child,parents):
        '''
        The fitted log-likelihood score for `child` having `parents`

        In addition to the score the number of joint instantations of the parents is returned.
        If this number would cause an overflow `None` is returned instead of the number.

        Args:
         child (str): The child variable
         parents (iter): The parent variables

        Returns:
         tuple: 
            (1) The fitted log-likelihood local score for the given family and 
            (2) the number of joint instantations of the parents (or None if too big)
        '''
        pah, numinsts = self.entropy(parents)
        return self._data_length * (pah - self.entropy(list(parents)+[child])[0]), numinsts

class DiscreteLL(AbsDiscreteLLScore):

    def __init__(self,data):
        '''Initialises a `DiscreteLL` object.

        Args:
         data (DiscreteData): data
        '''
        _AbsLLPenalised.__init__(self,data)



    def score(self,child,parents):
        '''
        Return the fitted log-likelihood score for `child` having `parents`, 
        and also upper bound on the score for proper supersets of `parents`.

        Args:
         child (str): The child variable
         parents (iter): The parent variables

        Returns:
         tuple: The fitted log-likelihood local score for the given family and an upper bound on the local score 
         for proper supersets of `parents`
        '''
        return self.ll_score(child,parents)[0], self._maxllh[child]

class DiscreteBIC(AbsDiscreteLLScore):

    def __init__(self,data,k=1):
        '''Initialises a `DiscreteBIC` object.

        Args:
         data (DiscreteData): data
         k (float): Multiply standard BIC penalty by this amount, so increase for sparser networks
        '''
        _AbsLLPenalised.__init__(self,data)
        fn = 0.5 * log(self._data_length)  # Carvalho notation
        self._child_penalties = {v:k*fn*(self.arity(v)-1) for v in self._variables}


class DiscreteAIC(AbsDiscreteLLScore):

    def __init__(self,data,k=1):
        '''Initialises an `DiscreteAIC` object.

        Args:
         data (DiscreteData): data
         k (float): Multiply standard AIC penalty by this amount, so increase for sparser networks
        '''
        _AbsLLPenalised.__init__(self,data)
        self._child_penalties = {v:k*(self.arity(v)-1) for v in self._variables}

class AbsGaussianLLScore(ContinuousData):
    """
    Abstract class for Gaussian log-likelihood scoring
    """


    def gaussianll(self,variables):
        '''
        Compute the Gaussian log-likelihood of some variables

        Args:
         variables (iter): Variables

        Returns:
         The Gaussian log-likelihood of some variables
        '''
        vset = frozenset(variables)
        try:
            return self._gaussianll_cache[vset]
        except KeyError:
            indices = [self._varidx[v] for v in variables]
            subcov = self._cov[np.ix_(indices,indices)]
            gll = -0.5 * self._data_length * (np.linalg.slogdet(subcov)[1] + len(indices)*self._log2pi1)
            self._gaussianll_cache[vset] = gll
            return gll
    
    def ll_score(self,child,parents):
        '''The Gaussian log-likelhood score for a given family, plus the number of free parameters

        Args:
         child (str): The child variable
         parents (iter) : The parents
       
        Returns:
         tuple: First element of tuple is the Gaussian log-likelihood score for the family for current data
          Second element is number of free parameters which is number of parents plus 1 (for intercept)
        '''
        return (self.gaussianll(list(parents)+[child]) - self.gaussianll(parents)), len(parents)+1

class GaussianLL(AbsGaussianLLScore):
    
    def __init__(self,data):
        '''Initialises an `GaussianLL` object.

        Args:
         data (ContinuousData): data
        '''
        _AbsLLPenalised.__init__(self,data)

    def score(self,child,parents):
        '''
        Return the fitted log-likelihood score for `child` having `parents`, 
        and also upper bound on the score for proper supersets of `parents`.

        Args:
         child (str): The child variable
         parents (iter): The parent variables

        Returns:
         tuple: The fitted log-likelihood local score for the given family and an upper bound on the local score 
         for proper supersets of `parents`
        '''
        return self.ll_score(child,parents)[0], self._maxllh[child]

        
class GaussianBIC(AbsGaussianLLScore):

    def __init__(self,data,k=1,sdresidparam=True):
        '''Initialises an `GaussianBIC` object.

        Args:
         data (ContinuousData): data
         k (float): Multiply standard BIC penalty by this amount, so increase for sparser networks
         sdresidparam (bool): Whether to count the standard deviation of the residuals as a parameter
          when computing the penalty
        '''
        _AbsLLPenalised.__init__(self,data)
        self._fn = k * 0.5 * log(self._data_length)  # Carvalho notation
        self._sdresidparam = sdresidparam
        
    def score(self,child,parents):
        '''
        Return the fitted Gaussian BIC score for `child` having `parents`, 
        and also upper bound on the score for proper supersets of `parents`.

        Args:
         child (str): The child variable
         parents (iter): The parent variables

        Returns:
         tuple: The Gaussian BIC local score for the given family and an upper bound on the local score 
         for proper supersets of `parents`
        '''
        this_ll_score, numparams = self.ll_score(child,parents)
        if self._sdresidparam:
            numparams += 1
        return this_ll_score - self._fn * numparams, self._maxllh[child] - self._fn * (numparams+1)

class GaussianAIC(AbsGaussianLLScore):

    def __init__(self,data,k=1,sdresidparam=True):
        '''Initialises an `GaussianAIC` object.

        Args:
         data (ContinuousData): data
         k (float): Multiply standard AIC penalty by this amount, so increase for sparser networks
         sdresidparam (bool): Whether to count the standard deviation of the residuals as a parameter
          when computing the penalty

        '''
        _AbsLLPenalised.__init__(self,data)
        self._k = k
        self._sdresidparam = sdresidparam

    def score(self,child,parents):
        '''
        Return the fitted Gaussian AIC score for `child` having `parents`, 
        and also upper bound on the score for proper supersets of `parents`.

        Args:
         child (str): The child variable
         parents (iter): The parent variables

        Returns:
         tuple: The Gaussian AIC local score for the given family and an upper bound on the local score 
         for proper supersets of `parents`
        '''
        this_ll_score, numparams = self.ll_score(child,parents)
        if self._sdresidparam:
            numparams += 1
        return this_ll_score - self._k * numparams, self._maxllh[child] - self._k * (numparams+1)

class GaussianL0(AbsGaussianLLScore):
    '''
    Implements score discussed in "l_0-Penalized Maximum Likelihood for Sparse Directed
    Acyclic Graphs" by Sara van de Geer and Peter Buehlmann. Annals of Statistics 41(2):536-567, 2013.
    '''
    def __init__(self,data,k=1):
        '''Initialises an `GaussianL0` object.

        Args:
         data (ContinuousData): data
         k (float): Tuning parameter for L0 penalty. Called "lambda^2" in van de Geer and Buehlmann
        '''
        _AbsLLPenalised.__init__(self,data)
        self._k = k

    def score(self,child,parents):
        '''
        Return the fitted Gaussian AIC score for `child` having `parents`, 
        and also upper bound on the score for proper supersets of `parents`.

        Args:
         child (str): The child variable
         parents (iter): The parent variables

        Returns:
         tuple: The Gaussian AIC local score for the given family and an upper bound on the local score 
         for proper supersets of `parents`
        '''
        this_ll_score, numparams = self.ll_score(child,parents)
        sb = numparams-1 # just count edges, so remove count for intercept
        return this_ll_score - self._k * sb, self._maxllh[child] - self._k * (sb+1)


# END Classes for penalised log-likelihood 
    
class BDeu(DiscreteData):
    """
    Discrete data with attributes and methods for BDeu scoring
    """

    def __init__(self,data,alpha=1.0):
        '''Initialises a `BDeu` object.

        Args:
         data (DiscreteData): data
         
         alpha (float): The *equivalent sample size*
        '''
        self.__dict__.update(data.__dict__)
        self.alpha = alpha
        self._cache = {}

        # for upper bounds
        self._atoms = get_atoms(self._data,self._arities)

        
    @property
    def alpha(self):
        '''float: The *equivalent sample size* used for BDeu scoring'''
        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        '''Set the *equivalent sample size* for BDeu scoring
        
        Args:
         alpha (float): the *equivalent sample size* for BDeu scoring

        Raises:
         ValueError: If `alpha` is not positive
        '''
        if not alpha > 0:
            raise ValueError('alpha (equivalent sample size) must be positive but was give {0}'.format(alpha))
        self._alpha = alpha

    def clear_cache(self):
        '''Empty the cache of stored BDeu component scores

        This should be called, for example, if new scores are being computed
        with a different alpha value
        '''
        self._cache = {}
        
    def upper_bound_james(self,child,parents,alpha=None):
        """
        Compute an upper bound on proper supersets of parents

        Args:
         child (str) : Child variable.
         parents (iter) : Parent variables
         alpha (float) : ESS value for BDeu score. If not supplied (=None)
          then the value of `self.alpha` is used.

        Returns:
         float : An upper bound on the local score for parent sets
         for `child` which are proper supersets of `parents`

        """
        if alpha is None:
            alpha = self._alpha
        child_idx = self._varidx[child]
        pa_idxs = sorted([self._varidx[v] for v in parents])
        for pa_idx in pa_idxs:
            alpha /= self._arities[pa_idx]
        r = self._arities[child_idx]

        # each element of atoms_ints is a tuple of ints:
        # (fullinst,childvalcounts,sum(childvalcounts),ok_first)
        # each element of atoms_floats is:
        # sum_n n*log(n/tot), where sum is over childvalcounts
        # and tot = sum(childvalcounts)
        atoms_ints, atoms_floats = self._atoms[0][child_idx], self._atoms[1][child_idx]

        if len(atoms_floats) == 0:
            return 0.0

        # remove cols corresponding to non-parents and order
        p = len(self._arities)
        end_idxs = list(range(p,p+r+2))
        atoms_ints_redux = atoms_ints[:,pa_idxs+end_idxs]
        if len(pa_idxs) > 0:
            idxs = np.lexsort([atoms_ints_redux[:,col] for col in range(len(pa_idxs))])
        else:
            idxs = list(range(len(atoms_floats)))

        return upper_bound_james_fun(atoms_ints_redux,atoms_floats,len(pa_idxs),alpha,r,idxs)
        
    def bdeu_score_component(self,variables,alpha=None):
        '''Compute the BDeu score component for a set of variables
        (from the current dataset).

        The BDeu score for a child v having parents Pa is 
        the BDeu score component for Pa subtracted from that for v+Pa
        
        Args:
         variables (iter) : The names of the variables
         alpha (float) : The effective sample size parameter for the BDeu score.
          If not supplied (=None)
          then the value of `self.alpha` is used.

        Returns:
         float : The BDeu score component.
        '''
        if alpha is None:
            alpha = self._alpha

        if len(variables) == 0:
            return lgamma(alpha) - lgamma(alpha + self._data_length), 1
        else:
            cols = np.array(sorted([self._varidx[x] for x in list(variables)]), dtype=np.uint32)
            return compute_bdeu_component(
                self._unique_data,self._unique_data_counts,cols,
                np.array([self._arities[i] for i in cols], dtype=self._arity_type),
                alpha,self._maxflatcontabsize)

    def _bdeu_score_component_cache(self,s):
        s_set = frozenset(s)
        try:
            score_non_zero_count = self._cache[s_set]
        except KeyError:
            score_non_zero_count = self.bdeu_score_component(s_set)
            self._cache[s_set] = score_non_zero_count
        return score_non_zero_count

    def bdeu_score(self, child, parents):

        parent_score, _ = self._bdeu_score_component_cache(parents)
        family_score, non_zero_count = self._bdeu_score_component_cache((child,)+parents)

        simple_ub = -log(self.arity(child)) * non_zero_count

        #james_ub = self.upper_bound_james(child,parents)
        
        #return parent_score - family_score, min(simple_ub,james_ub)
        return parent_score - family_score, simple_ub

        
    def bdeu_scores(self,palim=None,pruning=True,alpha=None):
        """
        Exhaustively compute all BDeu scores for all child variables and all parent sets up to size `palim`.
        If `pruning` delete those parent sets which have a subset with a better score.
        Return a dictionary dkt where dkt[child][parents] = bdeu_score
        
        Args:
         palim (int/None) : Limit on parent set size
         pruning (bool) : Whether to prune
         alpha (float) : ESS for BDeu score. 
          If not supplied (=None)
          then the value of `self.alpha` is used.



        Returns:
         dict : dkt where dkt[child][parents] = bdeu_score
        """
        if alpha is None:
            alpha = self._alpha
        
        if palim == None:
            palim = self._arities.size - 1

        score_dict = {}
        # Initialisation
        # Need to create dict for every child
        # also its better to do the zero size parent set calc here
        # so that we don't have to do a check for every parent set
        # to make sure it is not of size 0 when calculating score component size
        no_parents_score_component = lgamma(alpha) - lgamma(alpha + self._data_length)
        for c, child in enumerate(self._variables):
            score_dict[child] = {
                frozenset([]):
                no_parents_score_component
            }
        
        for pasize in range(1,palim+1):
            for family in combinations(self._variables,pasize): 
                score_component = self.bdeu_score_component(family,alpha)
                family_set = frozenset(family)
                for child in self._variables:
                    if child in family_set:
                        parent_set = family_set.difference([child])
                        score_dict[child][parent_set] -= score_component
                        if pruning and prune_local_score(score_dict[child][parent_set],parent_set,score_dict[child]):
                            del score_dict[child][parent_set]
                    else:
                        score_dict[child][family_set] = score_component 
                
        # seperate loop for maximally sized parent sets
        for vars in combinations(self._variables,palim+1):
            score_component = self.bdeu_score_component(vars,alpha)
            vars_set = frozenset(vars)
            for child in vars:
                parent_set = vars_set.difference([child])
                score_dict[child][parent_set] -= score_component
                if pruning and prune_local_score(score_dict[child][parent_set],parent_set,score_dict[child]):
                    del score_dict[child][parent_set]

            
        return score_dict

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




    
    
class MixedLL(MixedData):
    def __init__(self, data, header=True, comments='#', delimiter=None, standardise=False):
        """
        Initialize the Mixed Log-Likelihood class

        Parameters:
            data: str or MixedData
                Path to the .dat file or an existing MixedData instance.
            header: bool
                Whether the file has a header row.
            comments: str
                Character indicating comment lines.
            delimiter: str
                Field delimiter in the data file.
            standardise: bool
                Whether to standardize continuous variables.
        """
        
        # Initialize MixedData, which processes the file
        super().__init__(data, header=header, comments=comments, delimiter=delimiter, standardise=standardise)   
                    
        self.parent_threshold = 0.5
    

        # Create instances of DiscreteLL and GaussianLL using the separated datasets
        self.discrete_ll = DiscreteLL(self.discrete_data)
        self.gaussian_ll = GaussianLL(self.continuous_data)

        # Create a boolean mask for discrete variables
        self.discrete_mask = np.array([i in self.discrete_cols for i in range(len(self._variables))])
        
        # Compute median and scale (b) for Laplace distribution
        self._median = np.median(data._data, axis=0)
        self._b = np.mean(np.abs(data._data - self._median), axis=0) / np.log(2)


    def log_pi(self, val, idx):
        """
        Compute the log of the Laplace density function.
        """
        log_laplace = -np.log(2 * self._b[idx]) - np.abs(val - self._median[idx]) / self._b[idx]
        #print(f"Computing log_pi for value {val} at index {idx} with median {self._median[idx]} and scale {self._b[idx]} output: {log_laplace}")
        return log_laplace

    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def compute_log_likelihood(self, node_idx, conn_matrix, lambda_sparse=5):
        """
        Computes the negative log-likelihood for mixed data.

        Returns:
            float: The negative log-likelihood score for mixed data
        """
        n_samples = self._data.shape[0]
        n_nodes = len(self._variables)
        nll = 0.0  # negative log-likelihood accumulator
        
        
        # Count parents of node_idx for sparsity regularization
        num_parents = np.count_nonzero(conn_matrix[node_idx])
        
        for row in range(n_samples):

            x_i_t = self._data[row][node_idx]
            parent_sum = 0

            # Compute weighted sum from parents
            for parent_node in range(n_nodes):
                x_k_t = self._data[row][parent_node]
                parent_sum += conn_matrix[node_idx][parent_node] * x_k_t

            if self._is_discrete[node_idx]:
                parent_sum += self.biases[node_idx]
                prob = self.sigmoid(parent_sum)

                # Clip to prevent log(0)
                prob = np.clip(prob, 1e-10, 1 - 1e-10)

                # Bernoulli log-likelihood
                ll = x_i_t * np.log(prob) + (1 - x_i_t) * np.log(1 - prob)
                nll -= ll
            else:
                # Laplace log-density of residual: x_i_t - linear_pred
                residual = x_i_t - parent_sum
                ll = self.log_pi(residual, node_idx)
                nll += ll
                
        # Add sparsity regularizer penalty (L0 regularizer)
        nll += lambda_sparse * num_parents

        return -nll
    
    # def fit_parameters(self, X, y, is_discrete, lambda_val=0.1):
    #     """
    #     Fits a single parameter value for a parent variable using either
    #     logistic regression (discrete) or LAD regression (continuous).
        
    #     Args:
    #         X (ndarray): shape (n_samples,) or (n_samples, 1)
    #         y (ndarray): shape (n_samples,), target variable
    #         is_discrete (bool): True for classification, False for regression
    #         lambda_val (float): regularization strength

    #     Returns:
    #         float: the best-fit parameter value (scalar b_ij)
    #     """
    #     if X is None or X.size == 0:
    #         return 0.0  # no parent → no connection → weight is zero

    #     if X.ndim == 1:
    #         X = X.reshape(-1, 1)

    #     if is_discrete:
    #         model = LogisticRegression(
    #             penalty='l1' if lambda_val > 0 else 'none',
    #             C=1/lambda_val if lambda_val > 0 else 1e10,
    #             fit_intercept=True,
    #             solver='liblinear'
    #         )
    #         model.fit(X, y)
    #         print(f"Weight for discrete variable: {model.coef_[0][0]}")
    #         return model.coef_[0][0]  # just the weight for this parent

    #     else:
    #         n_samples = len(y)
    #         A_eq = np.hstack([X, -np.eye(n_samples), np.eye(n_samples)])
    #         b_eq = y
    #         c = np.concatenate([
    #             lambda_val * np.ones(1),
    #             np.ones(n_samples),
    #             np.ones(n_samples)
    #         ])
    #         bounds = [(-np.inf, np.inf)] + [(0, np.inf)] * (2 * n_samples)

    #         res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    #         if not res.success:
    #             raise RuntimeError(f"Linear programming failed: {res.message}")

    #         weight = res.x[0]
    #         print(f"Weight for continuous variable: {weight}")
    #         return weight

    def fit_parameters_multi(self, X, y, is_discrete, lambda_val=0.1):
        """
        Fits parameter values for a full parent set using logistic or LAD regression.

        Args:
            X (ndarray): shape (n_samples, n_parents)
            y (ndarray): shape (n_samples,)
            is_discrete (bool): True for classification, False for regression
            lambda_val (float): regularization strength

        Returns:
            np.ndarray: array of fitted weights (shape: n_parents,)
        """
        if X is None or X.size == 0:
            return np.zeros((0,))

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if is_discrete:
            model = LogisticRegression(
                penalty='l1' if lambda_val > 0 else 'none',
                C=1/lambda_val if lambda_val > 0 else 1e10,
                fit_intercept=True,
                solver='liblinear'
            )
            model.fit(X, y)
            print("Weights for discrete variable:", model.coef_[0])
            return model.coef_[0]  # shape: (n_parents,)

        else:
            # LAD regression via linear programming
            n_samples, n_parents = X.shape
            A_eq = np.hstack([X, -np.eye(n_samples), np.eye(n_samples)])
            b_eq = y
            c = np.concatenate([
                lambda_val * np.ones(n_parents),
                np.ones(n_samples),
                np.ones(n_samples)
            ])
            bounds = [(-np.inf, np.inf)] * n_parents + [(0, np.inf)] * (2 * n_samples)

            res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

            if not res.success:
                raise RuntimeError(f"LAD Linear programming failed: {res.message}")

            weights = res.x[:n_parents]
            print("Weights for continuous variable:", weights)
            return weights


    def score(self, child, parents):
        node_idx = self._varidx[child]
        parent_idxs = [self._varidx[p] for p in parents]

        X = self._data[:, parent_idxs]
        y = self._data[:, node_idx]

        weights = self.fit_parameters_multi(X, y, self._is_discrete[node_idx], lambda_val=0.1)

        conn_matrix = np.zeros((len(self._variables), len(self._variables)))
        for i, parent_idx in enumerate(parent_idxs):
            conn_matrix[node_idx][parent_idx] = weights[i]

        print(f"Connection matrix for {child} with parents {parents}:")
        print(conn_matrix)
        final_score = self.compute_log_likelihood(node_idx, conn_matrix)
        print(f"Final score for {child} with parents {parents}: {final_score}")
        return final_score, None


    # def score(self, child, parents):
    #     node_idx = self._varidx[child]
        
    #     conn_matrix = np.zeros((len(self._variables), len(self._variables)))
    #     for i in range(len(parents)):
    #         conn_matrix[node_idx][self._varidx[parents[i]]] = self.fit_parameters(
    #             self._data[:, self._varidx[parents[i]]], self._data[:, node_idx], self._is_discrete[node_idx], 0.1
    #         )
        
    #     final_score = self.compute_log_likelihood(node_idx, conn_matrix)
    #     print(f"Final score for {child} with parents {parents}: {final_score}")
    #     return final_score, None
        

    
    



    # def compute_log_likelihood(self):
    #     """
    #     Computes the negative log-likelihood for mixed data.

    #     Returns:
    #         float: The negative log-likelihood score for mixed data
    #     """
        
   
    #     for row in range(len(self._data)):
    #         for node in range(len(self._variables)):
    #             if self._is_discrete[node]:
    #                 # TODO
    #                 pass
    #             else:
    #                 node_parent_sum = 0
    #                 for parent_node in range(len(self._variables)):
    #                     # Compute score
    #                     node_parent_sum += self.conn_matrix[node][parent_node] * self._data(parent_node, row)
    #                 local_score = self.log_pi(self._data(node, row) - node_parent_sum)
        
    #     #TODO sum local_scores
    

        