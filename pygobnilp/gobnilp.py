#!/usr/bin/env python
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
   Python version of GOBNILP
"""

__author__ = "James Cussens"
__email__ = "james.cussens@york.ac.uk"


from itertools import combinations, permutations
import sys
from math import log
import warnings

try:
    from gurobipy import Model, LinExpr, GRB
except ImportError as e:
    print("Gurobi not available!")
    print(e)

try:
    import matplotlib.pyplot as plt
except ImportError as e:
    print("Could not import matplotlib.pyplot!")
    print(e)

try:
    from .discrete_scoring import DiscreteData, BDeu
except ImportError as e:
    print("Could not import BDeu score generating code!")
    print(e)

try:
    from .continuous_scoring import ContinuousData, BGe
except ImportError as e:
    print("Could not import BGe score generating code!")
    print(e)

try:
    import networkx as nx
    from networkx.algorithms.moral import moral_graph
except ImportError as e:
    print("Could not import networkx!")
    print(e)
    
try:
    import pandas as pd
except ImportError as e:
    print("Could not import pandas!")
    print(e)

try:
    import numpy as np
except ImportError as e:
    print("Could not import numpy!")
    print(e)

warnings.filterwarnings('ignore')


# utility functions

def _enforce_palim(dkt,palim):
    for scored_parentsets in dkt.values():
        for parentset in scored_parentsets.keys():
            if len(parentset) > palim:
                del scored_parentsets[parentset]


def from_bnlearn_modelstring(modelstring):
    '''Return a DAG from a bnlearn modelstring

    Args:
     modelstring (str) : A bnlearn modelstring defining a DAG

    Returns: 
     networkx.DiGraph : The DAG as a networkx Digraph
    '''
    bn = nx.DiGraph()
    for f in modelstring[1:-1].split(']['):
        x = f.split('|')
        ch = x[0]
        bn.add_node(ch)
        if len(x) == 2:
            for pa in x[1].split(':'):
                bn.add_edge(pa,ch)
    return bn

def read_local_scores(f,verbose=False):
    '''Read local scores from a named file, standard input or a file object,
    and return a dictionary dkt where dkt[child][parentset] is the local score 
    for the family child<-parentset

    The file is assumed to be in "Jaakkola" format.

    Args:
        f (str/file object) : The file containing the local scores. 

    Returns: 
     dict : Dictionary containing local scores
    '''

    if type(f) == str:
        if f == '-':
            f = sys.stdin
        else:
            f = open(f)

    family_scores = {}
    n = int(f.readline())
    if verbose:
        print('Problem has {0} variables'.format(n), file=sys.stderr)
    fields = f.readline().rstrip().split()
    def init(fields):
        # no check that there are only two fields
        return fields[0], int(fields[1]), 0, {} 
    current_variable, nscores, i, this_dkt = init(fields)
    for line in f:
        fields = line.rstrip().split()
        if i < nscores:
            # don't bother checking that fields[1] correctly specifies
            # the number of parents
            this_dkt[frozenset(fields[2:])] = float(fields[0])
            i += 1
        else:
            family_scores[current_variable] = this_dkt
            current_variable, nscores, i, this_dkt = init(fields)
    f.close()
    family_scores[current_variable] = this_dkt
    if verbose:
        print('Scores read in', file=sys.stderr)
    return family_scores


def _write_local_scores(dkt,f):
    if type(f) == str:
        if f == '-':
            f = sys.stdout
        else:
            f = open(f,'w')

    print(len(dkt),file=f)
    for child, scored_parentsets in dkt.items():
        print('{0} {1}'.format(child,len(scored_parentsets)),file=f)
        skores = [(score, parentset) for parentset, score in scored_parentsets.items()]
        skores.sort(reverse=True) # ensure local scores are printed from best to worst
        for score, parentset in skores:
            print('{0} {1} {2}'.format(score,len(parentset),' '.join(sorted(parentset))),file=f)
    f.close()

    
class MN(nx.Graph):

    def satisfy_ci(self,a,b,s):
        '''Does the Markov network satisfy `a` _|_ `b` | `s`?
        i.e. is there a path from a node in `a` to a node in
        `b` which avoids nodes in `s`?
        '''
        g = self.copy()
        g.remove_nodes_from(s)
        if len(a) > len(b):
            # want to iterate over smaller set
            a,b = b,a
        a_component = set()
        bset = frozenset(b)
        for node in a:
            if node not in a_component:
                a_component.update(nx.node_connected_component(g,node))
                if not a_component.isdisjoint(bset):
                    return False
        return True

class BN(nx.DiGraph):
    '''
    Subclass of 
    `networkx.DiGraph <https://networkx.github.io/documentation/stable/reference/classes/digraph.html>`_. 
    See documentation for 
    `networkx.DiGraph <https://networkx.github.io/documentation/stable/reference/classes/digraph.html>`_ 
    for all methods not documented here.

    At present this class only implements the structure of a BN - the DAG.
    '''
    def __str__(self):
        '''
        Returns a textual representation of the BN

        Returns: 
         str: A textual representation of the BN
        '''
        res = '**********\nBN has score {0}\n**********\n'.format(self.graph['score'])
        for child, local_score in self.nodes(data='local_score'):
            res += '{0}<-{1} {2}\n'.format(child, ','.join(sorted(self.predecessors(child))), local_score)
        res += '**********\n'
        res += 'bnlearn modelstring = \n'
        res += self.bnlearn_modelstring()
        return res

    def plot(self):
        '''
        Generate and show a plot of the BN structure (DAG)
        '''
        nx.draw_networkx(self,pos=nx.drawing.nx_agraph.graphviz_layout(self,prog='dot'),
                         node_color="white",arrowsize=15)
        plt.show()

    def bnlearn_modelstring(self):
        '''Return a string representation suitable for bnlearn's "modelstring" function
        
        Returns:
         str: A string representation of the BN structure (DAG) suitable for bnlearn's "modelstring" function
        '''
        mstr = ''
        for child in self.nodes:
            parents = ':'.join(self.predecessors(child))
            mstr += '[{0}{1}{2}]'.format(child,'|' if parents else '',parents)
        return mstr

    def minimal_ancestral_graph(self,nodes):
        '''Find the minimal ancestral graph of self
        containing `nodes`

        Args:
         nodes (iter): The nodes for the minimal ancestral graph

        Returns:
         network.DiGraph: The minimal ancestral graph
        '''
        ancestors = set()
        for node in nodes:
            if node not in ancestors:
                ancestors.update(nx.ancestors(self,node))
            ancestors.add(node)
        return self.subgraph(ancestors)
    
    def satisfy_ci(self,a,b,s):
        '''Does the DAG satisfy this conditional independence relation\\:
        `a` is independent of `b` conditional on `s`?

        Args:
         a (iter) : A set of BN variables
         b (iter) : A set of BN variables
         s (iter) : A set, possibly empty, of BN variables

        Returns:
         tuple: A pair where the first element is a bool stating whether
         the given conditional independence relation is satisfied and the second
         is the minimal ancestral graph containing `a`, `b` and `s`.
        '''
        mag = self.minimal_ancestral_graph(set().union(a,b,s))
        return MN(moral_graph(mag)).satisfy_ci(a,b,s), mag.nodes
        
    def connected(self,u,v):
        '''Are `u` and `v` connected (in either direction)
        
        Args:
         u (str) : A node
         v (str) : A node

        Returns:
         bool: Whether `u` and `v` are connected
        '''
        return self.has_edge(u,v) or self.has_edge(v,u)

    def adjacency_matrix(self):
        '''The adjacency matrix

        Returns:
         numpy.matrix: The adjacency matrix
        '''
        return nx.to_numpy_matrix(self)
    
    def cpdag(self,compelled=()):
        '''Return the CPDAG representation of the Markov equivalence class of the input 
        DAG
        
        Starting from a initial set of edges whose direction is *compelled* to be that
        given in the DAG, 
        the following 3 rules from Koller and Friedman (here shown in Prolog) are used::

         %R1
         compelled(Y,Z) :- edge(Y,Z), compelled(X,Y), not edge(X,Z), not edge(Z,X).
         %R2
         compelled(X,Z) :- edge(X,Z), compelled(X,Y), compelled(Y,Z).
         %R3
         compelled(X,Z) :- edge(X,Z), compelled(Y1,Z), compelled(Y2,Z), 
                           (edge(X,Y1);edge(Y1,X)),  (edge(X,Y2);edge(Y2,X)).

        This method uses the "semi-naive evaluation" algorithm for computing 
        the relevant least Herbrand model, which is the set of all the compelled edges.

        Args:
         compelled (iter): Edges to set as compelled in addition to those involved in
          *immoralities*. 
        
        Returns:
         gobnilp.CPDAG : the CPDAG representation of the Markov equivalence class of the input 
         DAG
        '''

        new_compelled = list(compelled)
        num_uncompelled_parents = {}
        for v in self.nodes():
            num_uncompelled_parents[v] = self.in_degree(v)
            for pa1, pa2 in combinations(self.predecessors(v),2):
                if not self.connected(pa1,pa2):
                    new_compelled.append((pa1,v))
                    new_compelled.append((pa2,v))

        compelled = set()
        above_two_parents = [z for z in self.nodes() if self.in_degree[z] > 2] 
        while new_compelled != []:

            compelled.update(new_compelled)
            for pa, ch in new_compelled:
                num_uncompelled_parents[ch] -= 1                
            
            old_new_compelled = new_compelled[:]
            new_compelled = []

            for (x,y) in old_new_compelled:
                for z in self.successors(y):

                    if num_uncompelled_parents[z] == 0:
                        continue

                    if (y,z) not in compelled:
                        # R1
                        if not self.connected(x,z):
                            new_compelled.append((y,z))
                    else:
                        # R2
                        if self.has_edge(x,z):
                            new_compelled.append((x,z))

                (yy,zz) = (x,y) #renaming for R2
                if num_uncompelled_parents[zz] > 0:
                    for xx in self.predecessors(yy):
                        # R2
                        if (xx,yy) in compelled and self.has_edge(xx,zz):
                            new_compelled.append((xx,zz))

                (y1,zz) = (x,y) #renaming for R3
                if zz in above_two_parents and num_uncompelled_parents[zz] > 0:
                    tmp = set(self.predecessors(zz))
                    tmp.remove(y1)
                    for (xx,y2) in permutations(tmp,2):
                        # R3
                        if ((y2,zz) in compelled and
                            (xx,zz) not in compelled and
                            self.connected(xx,y1) and
                            self.connected(xx,y2)):
                            new_compelled.append((xx,zz))

                            
        cpdag = CPDAG(nx.to_dict_of_dicts(self))
        for edge in self.edges():
            cpdag.edges[edge]['compelled'] = (edge in compelled)
        return cpdag

class CPDAG(BN):
    '''Subclass of :py:class:`gobnilp.BN <gobnilp.BN>` which is itself a subclass of
    `networkx.DiGraph <https://networkx.github.io/documentation/stable/reference/classes/digraph.html>`_. 
    See documentation for those classes for all methods not documented here.
    '''

    directed_arrow_text = '->'
    '''Text to indicate a directed arrow'''
    undirected_arrow_text = '-'
    '''Text to indicate a undirected edge'''
    directed_arrow_colour = 'red'
    '''Colour to indicate a directed arrow'''
    undirected_arrow_colour = 'black'
    '''Colour to indicate a undirected edge'''
    
    _edge_arrow = {True:directed_arrow_text,False:undirected_arrow_text}
    _edge_colour = {True:directed_arrow_colour,False:undirected_arrow_colour}

    def __str__(self):
        '''
        Returns a textual representation of the CPDAG

        Returns: 
         str: A textual representation of the CPDAG
        '''

        res = '**********\n'
        res += 'CPDAG:\nVertices: {0}\n'.format(','.join(self.nodes))
        for (u,v,compelled) in self.edges.data('compelled'):
            res += '{0}{1}{2}\n'.format(u,self._edge_arrow[compelled],v)
        return res

    def plot(self,abbrev=True):
        '''
        Generate and show a plot of the CPDAG

        A DAG from the Markov equivalence class defined by the CPDAG is shown.
        Reversible and irreversible arrows are distinguished by colour. By default
        the colours are black and red, respectively.

        Args:
         abbrev (int) : Whether to abbreviate variable names to first 3 characters.
        '''
        if abbrev:
            ls = dict((x,x[:3]) for x in self.nodes)
        else:
            ls = None
        edge_colors = [self._edge_colour[compelled] for (u,v,compelled) in self.edges.data('compelled')]
        nx.draw_networkx(self,pos=nx.drawing.nx_agraph.graphviz_layout(self,prog='dot'),
                         node_color="white",arrowsize=15,edge_color=edge_colors,labels=ls)
        plt.show()

    
class Gobnilp(Model):
    '''Subclass of `the Gurobi Model class 
    <https://www.gurobi.com/documentation/8.1/refman/py_model.html>`_ specific 
    to learning Bayesian networks.
    See documentation for `the Gurobi Model class 
    <https://www.gurobi.com/documentation/8.1/refman/py_model.html>`_ 
    for all methods not documented here.
    '''

    allowed_user_constypes = [
        "forbidden_arrows","forbidden_adjacencies",
        "obligatory_arrows","obligatory_adjacencies",
        "obligatory_ancestors","forbidden_ancestors",
        "obligatory_conditional_independences"]

    # each stage indicates what is available at that stage
    stages = (
        'no data', 'data',  'local scores',
        'MIP model', 'MIP solution', 'BN(s)',
        'CPDAG(s)', 'output shown', 'output written')

    stages_set = frozenset(stages)
    
    stage_index = {stage:i for i, stage in enumerate(stages)}

    def between(self,stage1,stage2,stage3):
        '''Is stage2 strictly after stage1 but not strictly after stage 3
        '''
        return (self.stage_index[stage1] < self.stage_index[stage2] and
                self.stage_index[stage2] <= self.stage_index[stage3])

    def before(self,stage1,stage2):
        '''Is stage2 strictly after stage1
        '''
        return self.stage_index[stage1] < self.stage_index[stage2]

    @property
    def data_arities(self):
        '''numpy.array: Arities of the variables in the data
        
        The order of the arities matches the order of the variables 
        in the data, see: `data_variables` not the order in `bn_variables` 
        (which is always in sorted order).

        Raises AttributeError if continuous data being used

        '''
        return self._data.arities()

    @property
    def data_variables(self):
        '''list: the variables in the data
        
        Variables are in the order supplied by the original data source
        not the order in `bn_variables` 
        (which is always in sorted order).

        '''
        return self._data.variables()

    @property
    def data(self):
        '''pandas.DataFrame: Data associated with the instance
        '''
        return self._data.data()
    
    @property
    def rawdata(self):
        '''numpy.array: Raw data associated with the instance

        Returns a two-dimensional array with one row for each datapoint
        (and thus one colun for each variable).

        If the data is discrete the array entries are of dtype
        uint32 and if the data is discrete the entries are of dtype
        float64

        '''
        return self._data.rawdata()
    
    @property
    def learned_cpdags(self):
        '''tuple: Learned CPDAGs
        '''
        try:
            return self._learned_cpdags
        except AttributeError:
            raise Gobnilp.StageError(self.stage,"No CPDAGs learned/created yet.")            

        
    @property
    def learned_bns(self):
        '''tuple: Learned BNs
        '''
        try:
            return self._nxs
        except AttributeError:
            raise Gobnilp.StageError(self.stage,"No BNs learned yet.")            
        
    @property
    def learned_bn(self):
        '''BN: Learned BN (a maximally scoring one if several learned)
        '''
        return self.learned_bns[0]

    @property
    def learned_cpdag(self):
        '''CPDAG: Learned CPDAG (a maximally scoring one if several learned)
        '''
        return self.learned_cpdags[0]

    
    @property
    def stage(self):
        '''str: Stage of solving
        '''
        return "Gobnilp stage = {0}, Gurobi status = {1}".format(self._stage,self.Status)
        
    @property
    def n(self):
        '''int: The number of BN variables
        '''
        return len(self.bn_variables)

    @property
    def bn_variables(self):
        '''list: The BN variables (in order)'''
        return self._bn_variables

    # have to use method since property setting not working
    # due to Gobnilp being a subclass of Gurobi
    def set_bn_variables(self,bnvars):
        '''Set the BN variables to be a subset of existing BN variables
        
        Args:
         bnvars (iter): A subset of the existing BN variables 

        Raises:
         ValueError: If `bnvars` is not a subset of the variables in the data
        '''
        if not frozenset(bnvars).issubset(self._data.variables()):
            raise ValueError('{0} is not a subset of {1}'.format(bnvars,self._data.variables()))
        else:
            self._bn_variables = sorted(bnvars)
        
    @property
    def bn_variables_index(self):
        '''dict: Maps each BN variable to its index in the sorted list of BN variables'''
        return {v:i for i, v in enumerate(self.bn_variables)} 

    @property
    def ordered_parentsets(self):
        '''dict: For each child a list of parent sets sorted by local score

        Higher scoring parent sets come before lower scoring ones. Each parent set
        is a frozenset.

        Raises:
         Gobnilp.StageError: If no local scores have been created
        '''
        fs = self.family_scores
        os = {}
        for child, scored_parentsets in list(fs.items()):
            os[child] = sorted(scored_parentsets,key=lambda x: scored_parentsets[x],reverse=True)
        return os
    
    @property
    def family_scores(self):
        ''' dict: Dictionary of family scores (aka local scores)

        ``family_scores[child][parentset]`` is the local score for ``child`` having
        ``parentset`` (a frozenset) as its parents.

        Raises:
         Gobnilp.StageError: If no local scores have been created
        '''
        try:
            return self._family_scores
        except AttributeError:
            raise Gobnilp.StageError(self.stage,"No local scores computed yet.")

    local_scores = family_scores
    '''alternative name for family_scores
    '''
    
    @property
    def family(self):
        ''' dict: Dictionary of family variables
        (if these variables have been created, by default they are)

        Assuming the appropriate constraints have been added,
        ``family[child][parent_set]`` is the family variable indicating that ``parent_set`` is the
        parent set for ``child``.

        Raises:
         Gobnilp.StageError: If no family variables have been created
        '''
        return self._getmipvars('family')

    @property
    def family_list(self):
        '''list: ``family_list[i]`` is the family variable for the family with index ``i``.
        (if these variables have been created, by default they are)

        See also:
            :py:meth:`get_family_index <gobnilp.Gobnilp.get_family_index>`,
            :py:meth:`child <gobnilp.Gobnilp.child>`,
            :py:meth:`parents <gobnilp.Gobnilp.parents>`

        Raises:
         Gobnilp.StageError: If no family variables have been created
        '''
        return self._getmipvars('family_list')


    @property
    def arrow(self):
        ''' dict: Dictionary of arrow variables
        (if these variables have been created, by default they are)

        Assuming the appropriate constraints have been added,
        ``arrow[pa,ch]`` is the arrow variable indicating that there is an arrow from ``pa``
        to ``ch``.

        Raises:
         Gobnilp.StageError: If no arrow variables have been created
        '''
        return self._getmipvars('arrow')

    @property
    def adjacency(self):
        ''' dict: Dictionary of adjacency variables
        (if these variables have been created, by default they are)

        Assuming the appropriate constraints have been added,
        ``adjacency[{v1,v2}]`` is the adjacency variable indicating that ``v1`` and ``v2`` are adjacent.

        Raises:
         Gobnilp.StageError: If no adjacency variables have been created
        '''
        return self._getmipvars('adjacency')


    @property
    def total_order(self):
        ''' dict: Dictionary of total order variables
        (if these variables have been created, by default they are not)

        Assuming the appropriate constraints have been added,
        if it exists ``total_order[v1,v2]`` is the total order variable indicating that ``v1 > v2``.

        Raises:
         Gobnilp.StageError: If no total order variables have been created
        '''
        return self._getmipvars('total_order')

    @property
    def generation_index(self):
        ''' dict: Dictionary of generation index variables
        (if these variables have been created, by default they are not)

        Assuming the appropriate constraints have been added,
        if it exists ``generation_index[v1,pos]`` indicates whether ``v1`` has generation number ``pos``

        Raises:
         Gobnilp.StageError: If no generation index variables have been created
        '''
        return self._getmipvars('generation_index')

    
    @property
    def generation(self):
        ''' dict: Dictionary of generation variables
        (if these variables have been created by calling
        :py:meth:`add_variables_gen <gobnilp.Gobnilp.add_variables_gen>`)

        Assuming appropriate constraints have been added, by,
        for example, calling the method 
        :py:meth:`add_constraints_gen_arrow_indicator <gobnilp.Gobnilp.add_constraints_gen_arrow_indicator>`,
        then ``generation[v1]`` is the generation number for ``v1``.

        See Section 3.1.2 of `Maximum likelihood pedigree reconstruction using integer 
        programming (Cussens, 2010) <https://www.cs.york.ac.uk/ftpdir/pub/aig/Papers/james.cussens/wcb10.pdf>`_ 
        for details of how generation variables can be used to rule out cycles in directed graphs. 

        Raises:
         Gobnilp.StageError: If no generation variables have been created

        See also:
         * :py:meth:`add_variables_gen <gobnilp.Gobnilp.add_variables_gen>`
         * :py:meth:`add_constraints_gen_arrow_indicator <gobnilp.Gobnilp.add_constraints_gen_arrow_indicator>`
        '''
        return self._getmipvars('generation')

    @property
    def generation_difference(self):
        ''' dict: Dictionary of generation difference variables
        (if these variables have been created by calling
        :py:meth:`add_variables_gendiff <gobnilp.Gobnilp.add_variables_gendiff>`)

        Assuming the appropriate constraints have been added, by, for example,
        calling the method
        :py:meth:`add_constraints_gendiff <gobnilp.Gobnilp.add_constraints_gendiff>`,
        then ``generation_difference[v1,v2] = generation[v1] - generation[v2]``

        Raises:
         Gobnilp.StageError: If no generation difference variables have been created

        See also:
         * :py:meth:`add_variables_gendiff <gobnilp.Gobnilp.add_variables_gendiff>`
         * :py:meth:`add_constraints_gendiff <gobnilp.Gobnilp.add_constraints_gendiff>`
        '''
        return self._getmipvars('generation_difference')

    @property
    def absolute_generation_difference(self):
        ''' dict: Dictionary of absolute generation difference variables
        (if these variables have been created, by default they are not)

        Assuming the appropriate constraints have been added,
        if it exists ``absolute_generation_difference[v1,v2]`` is the absolute value of 
        ``generation_difference[v1,v2]``.
        
        Raises:
         Gobnilp.StageError: If no absolute generation difference variables have been created
        '''
        return self._getmipvars('absolute_generation_difference')
        
    @property
    def get_family_index(self):
        '''dict: Maps a family to its index

        ``get_family_index[child][parents]`` is the index for the given family.

        See also:
            :py:meth:`child <gobnilp.Gobnilp.child>` and 
            :py:meth:`parents <gobnilp.Gobnilp.parents>` and 
            :py:meth:`family_list <gobnilp.Gobnilp.family_list>`.
        '''
        return self._getidxdicts('get_family_index')


    @property
    def child(self):
        '''list: ``child[i]`` is the child in the family with index ``i``.

        See also:
            :py:meth:`get_family_index <gobnilp.Gobnilp.get_family_index>`,
            :py:meth:`parents <gobnilp.Gobnilp.parents>` and 
            :py:meth:`family_list <gobnilp.Gobnilp.family_list>`.
        '''
        return self._getidxdicts('child')

    @property
    def parents(self):
        '''list: ``parents[i]`` is the parent set in the family with index ``i``.

        The parent set is a frozenset.

        See also:
            :py:meth:`get_family_index <gobnilp.Gobnilp.get_family_index>`,
            :py:meth:`child <gobnilp.Gobnilp.child>`,
            :py:meth:`family_list <gobnilp.Gobnilp.family_list>`
        '''
        return self._getidxdicts('parents')

    @property
    def forbidden_arrows(self):
        return self._forbidden_arrows

    @property
    def obligatory_arrows(self):
        return self._obligatory_arrows

    @property
    def forbidden_adjacencies(self):
        return self._forbidden_adjacencies

    @property
    def obligatory_adjacencies(self):
        return self._obligatory_adjacencies

    @property
    def forbidden_ancestors(self):
        return self._forbidden_ancestors

    @property
    def obligatory_ancestors(self):
        return self._obligatory_ancestors

    @property
    def obligatory_conditional_independences(self):
        return self._obligatory_conditional_independences

    
    def __init__(self,name="gobnilp",verbose=0,gurobi_output=False,params=None):
        '''Initialise a Gobnilp object

        Args:
         name (str) : A name for the model
         verbose (int) : How much information to show when adding variables and constraints (and computing scores)
         gurobi_output (bool) : Whether to show output generated by Gurobi.
         params (str/None) : If not None, Gurobi parameter settings for parameters as a single string where each
          parameter setting is of the form ParamName=Value and different settings are separated by a space. For example,
          "TimeLimit=3 Heuristics=0.2 NodefileDir=tmpdir"
        '''
        super(Gobnilp,self).__init__(name)

        self._verbose = verbose

        self.Params.OutputFlag = gurobi_output
        
        # Setting Gurobi model parameters 
        self.ModelSense = -1      # maximise objective
        self.Params.PreCrush = 1  # since (always) adding cuts
        self.Params.CutPasses = 100000    # want to allow many cuts
        self.Params.GomoryPasses = 100000 # want to allow many cuts
        self.Params.MIPFocus = 2          # focus on proving optimality
        #self.Params.Presolve = 2
        #self.Params.PreDual = 2
        self.Params.ZeroHalfCuts = 2

        
        self.Params.MIPGap = 0
        self.Params.MIPGapAbs = 0

        # gobnilp parameters
        self._subip_cutoff = -0.999
        self._subip_cutting_timelimit = 5
        self._user_cuts_rounds_count = 0
        self._user_cuts_rounds_limit = None
        self._user_cuts_stalling_limit = None
        self._user_enforcement_rounds_count = 0
        self._user_enforcement_rounds_limit = None
        self._max_cluster_size = None
        
        self._cs = set()

        # remaining attributes will eventually get useful values
        # if they are needed
        self._idxdicts = {}

        # dictionary of dictionaries for MIP variables
        self._mipvars = {}

        self._c2 = None
        self._c3 = None

        self._enforcing_cluster_constraints = None
        self._enforcing_matroid_constraints = None
        self._adding_cluster_cuts = None
        self._adding_matroid_cuts = None

        # attributes for one dag per mec constraint
        self._one_dag_per_MEC = None
        self._MEC_constraint_dynamic = None
        self._MEC_constraint_careful = None
        self._mecrep = None

        self._last_bound = None

        # list of feasible solutions
        self._starts = []
        
        for constype in self.allowed_user_constypes:
            setattr(self,'_'+constype,set())

        if params is not None:
            for field in params.split():
                p,v = field.split('=')
                fn = self.getParamInfo(p)[1]
                self.setParam(p,fn(v))

        self._stage = 'no data'
                
    def _getmipvars(self,vtype):
        try:
            return self._mipvars[vtype]
        except KeyError:
            raise Gobnilp.StageError(self.stage,"No {0} variables in the model yet.".format(vtype))

    def add_forbidden_arrow(self,u,v):
        if (u,v) in self._forbidden_arrows:
            return
        if u == v:
            raise Gobnilp.UserConstraintError("Can't have {0}->{1}!".format(u,v))
        if (u,v) in self._obligatory_arrows:
            raise Gobnilp.UserConstraintError("Can't have {0}->{1} forbidden and obligatory!".format(u,v))
        if frozenset([u,v]) in self._obligatory_adjacencies:
            self.add_obligatory_arrow((v,u))
        self._forbidden_arrows.add((u,v))
            
    def add_obligatory_arrow(self,u,v):
        if (u,v) in self._obligatory_arrows:
            return
        if u == v:
            raise Gobnilp.UserConstraintError("Can't have {0}->{1}!".format(u,v))
        if (u,v) in self._forbidden_arrows:
            raise Gobnilp.UserConstraintError("Can't have {0}->{1} obligatory and forbidden!".format(u,v))
        self._obligatory_arrows.add((u,v))
        self.add_obligatory_adjacency(frozenset([u,v]))
        self.add_obligatory_ancestor(u,v)

    def add_forbidden_ancestor(self,u,v):
        if (u,v) in self._forbidden_ancestors:
            return
        if u == v:
            raise Gobnilp.UserConstraintError("Can't have {0}-->{1}!".format(u,v))
        if (u,v) in self._obligatory_ancestors:
            raise Gobnilp.UserConstraintError("Can't have {0}-->{1} forbidden and obligatory!".format(u,v))
        self.add_forbidden_arrow(u,v)
        self._forbidden_ancestors.add((u,v))

    def add_obligatory_ancestor(self,u,v):
        if (u,v) in self._obligatory_ancestors: #stop recursion safety!
            return
        if u == v:
            raise Gobnilp.UserConstraintError("Can't have {0}-->{1}!".format(u,v))
        if (u,v) in self._forbidden_ancestors:
            raise Gobnilp.UserConstraintError("Can't have {0}-->{1} obligatory and forbidden!".format(u,v))
        if (v,u) in self._obligatory_ancestors:
            raise Gobnilp.UserConstraintError("Can't have {0}-->{1} obligatory since {1}-->{0} is too!".format(u,v))
        self._obligatory_ancestors.add((u,v))
        # transitivity
        for (a,b) in self._obligatory_ancestors:
            if a == v:
                self.add_obligatory_ancestors(u,b)
            if b == u:
                self.add_obligatory_ancestors(a,v)

    def add_forbidden_adjacency(self,uv):
        uv = frozenset(uv)
        if uv in self._forbidden_adjacencies: 
            return
        u,v = tuple(uv)
        if u == v:
            raise Gobnilp.UserConstraintError("Can't have {0}-{1}!".format(u,v))
        if uv in self._obligatory_adjacencies:
            raise Gobnilp.UserConstraintError("Can't have {0}-{1} forbidden and obligatory!".format(u,v))
        self.add_forbidden_arrow(u,v)
        self.add_forbidden_arrow(v,u)
        self._forbidden_adjacencies.add(uv)
        
    def add_obligatory_adjacency(self,uv):
        uv = frozenset(uv)
        if uv in self._obligatory_adjacencies: 
            return
        u,v = tuple(uv)
        if u == v:
            raise Gobnilp.UserConstraintError("Can't have {0}-{1}!".format(u,v))
        if uv in self._forbidden_adjacencies:
            raise Gobnilp.UserConstraintError("Can't have {0}-{1} forbidden and obligatory!".format(u,v))
        self._obligatory_adjacencies.add(uv)
        if (u,v) in self._forbidden_arrows and (v,u) in self._forbidden_arrows:
            raise Gobnilp.UserConstraintError("Can't have {0}-{1} obligatory since both {0}->{1} and {1}->{0} are forbidden!".format(u,v))
        if (u,v) in self._forbidden_arrows:
            self.add_obligatory_arrow(v,u)
        if (v,u) in self._forbidden_arrows:
            self.add_obligatory_arrow(u,v)

    def add_obligatory_independence(self,a,b):
        self.add_obligatory_conditional_independence(a,b,frozenset())
        
    def add_obligatory_conditional_independence(self,a,b,s):
        aset = frozenset(a)
        bset = frozenset(b)
        sset = frozenset(s)
        if (aset,bset,sset) in self._obligatory_conditional_independences:
            return
        for x,y in [(aset,bset),(aset,sset),(bset,sset)]:
            if not x.isdisjoint(y):
                raise Gobnilp.UserConstraintError(
                    "In conditional independence constraint {0} and {1} must be disjoint.".format(x,y))
        for x in a:
            for y in b:
                self.add_forbidden_adjacency(frozenset([x,y]))
                if not s:
                    self.add_forbidden_ancestor(x,y)
                    self.add_forbidden_ancestor(y,x)

        # could do more propagation here
        
        self._obligatory_conditional_independences.add((aset,bset,sset))
            
    def set_start(self,dag):
        self.set_starts([dag])
        
    def set_starts(self,dags):
        for dag in dags:
            if type(dag) == str:
                # assume a 'model string'
                self._starts.append(from_bnlearn_modelstring(dag))
            elif isinstance(dag,nx.DiGraph):
                self._starts.append(dag)
            else:
                raise ValueError('{0} is not a DAG'.format(dag))
                
    def _set_mip_starts(self):
        '''Set a number of MIP start solutions
        '''
        dags = self._starts
        self.NumStart = len(dags)
        for i, dag in enumerate(dags):
            self.Params.StartNumber = i
            for child in dag.nodes:
                self.family[child][frozenset(dag.predecessors(child))].Start = 1
            
    def optimize(self):
        '''Solve the MIP model constructed by Gobnilp
        
        This overrides Gurobi's optimize method by hard coding in calls to
        a callback function for adding Gobnilp specific lazy constraints (and cuts).
        '''
        self._set_mip_starts()
        super(Gobnilp,self).optimize(lambda model, where: model._mycallback(where))

    
    def _getidxdicts(self,dtype):
        try:
            return self._idxdicts[dtype]
        except KeyError:
            raise Gobnilp.StageError(self.stage,"{0} not found. No family variables in the model yet.".format(dtype))

    
    def dag2mec(self,dag):
        '''Finds the Markov equivalence class for a DAG

        Args:
            dag (iterable): A DAG represented by a collection of 
             (indices for) 'families', where each family is a BN variable 
             together with its parents. 

        Returns:
            frozenset: A set of sets which is a sparse representation of the characteristic imset
            which represents the Markov equivalence class. 
            If (and only if) a set has value 1 in the characteristic imset then it is included as an element in the returned set.

        See also:
            Note that the BN variable, its parents and the binary MIP
            variable for a family can be recovered from its index using the methods
            :py:meth:`child <gobnilp.Gobnilp.child>`,
            :py:meth:`parents <gobnilp.Gobnilp.parents>` and 
            :py:meth:`family_variable <gobnilp.Gobnilp.family_variable>`,
            respectively.

            A rough 'inverse' of this method is :py:meth:`mec2dag <gobnilp.Gobnilp.mec2dag>`.
        '''
        c = set()
        child_list = self.child
        parentset_list = self.parents
        #print 'MEC for'
        for i in dag:
            child = child_list[i]
            parent_set = parentset_list[i]
            #print '{0}<-{1}'.format(child, ','.join(sorted(parent_set)))
            child_singleton = frozenset([child])
            for size in range(1,len(parent_set)+1):
                for subset in combinations(parent_set,size):
                    c.add(frozenset(subset) | child_singleton )
        #print 'is ', c
        return frozenset(c)

    def mec2dag(self,c,vs=None):
        '''Returns a DAG representative from a Markov equivalence class.

        Args:
            c (frozenset): A Markov equivalence class represented by a characteristic imset which is
              itself (sparsely) represented by the set of all sets with value 1 in the characteristic imset.
              This method assumes that `c` is a valid characteristic imset without any checking.
            vs (iterable): An ordering for the BN variables which determines the particular DAG representative
              returned. If ``None`` the ordering determined by Python ``sort`` is used.

        Returns:
            list/None: The DAG represented by a list of family indices, or None if the required families are not represented.

        See also:
            A rough 'inverse' of this method is :py:meth:`dag2mec <gobnilp.Gobnilp.dag2mec>`.
        '''
        fvs = []
        if vs is None:
            vs = self.bn_variables
        vs = list(vs)           # so as not to destroy the input!
        star = {}
        for v in vs:
            star[v] = set()
        for ci in c:
            for v in ci:
                star[v].add(ci)
        while vs != []:
            for v in tuple(vs): # work on a copy
                max_size = 0
                nbrs = set()
                biggest = frozenset()
                # find all vertices adjacent to v and
                # find biggest c-imset component containing v
                for ci in star[v]:
                    if len(ci) == 2:
                        nbrs.update(ci)
                    if len(ci) > max_size:
                        biggest = ci
                        max_size = len(ci)
                if frozenset(nbrs) == biggest:
                    # we can make all adjacent vertices into parents
                    # without alterning c-imset representation
                    fvs.append((v,biggest - frozenset([v])))
                    # remove v and the relevant c-imset components                
                    vs.remove(v)
                    for ci in star[v]:
                        for v2 in ci:
                            if v2 != v:
                                star[v2].remove(ci)
                    del star[v]
                    break
            else:
                raise ValueError("Couldn't find a sink.")
        get_family_index = self.get_family_index
        try:
            return [get_family_index[family] for family in fvs] 
        except KeyError:
            return None

    def _return_pruned_local_scores(self,local_score,palim):

        verbose = self._verbose
        bn_variables = self.bn_variables
        bn_variablesset = frozenset(bn_variables)
        
        score_dkt = {}
        search_nodes_expanded = 0
        scores_computed = 0

        for child in bn_variables:

            forbidden_parents = frozenset([child]+[parent for (parent,x) in self.forbidden_arrows if x == child])
            obligatory_parents = frozenset([parent for (parent,x) in self.obligatory_arrows if x == child])
            forbidden_pairs = frozenset(
                [frozenset((x,y)) for (a,b,s) in self.obligatory_conditional_independences if child in s for x in a for y in b])

            unfixed_parents = sorted(bn_variablesset - forbidden_parents - obligatory_parents)
            thisvaridx = {v:i for i, v in enumerate(unfixed_parents)}
            thispalim = palim - len(obligatory_parents)
            obligatory_parents_tuple = tuple(obligatory_parents)
            
            if thispalim < 0:
                raise Gobnilp.UserConstraintError("Variable {0} has {1} as obligatory parents but the parent set size limit is set to {2}".
                                                  format(child,','.join(sorted(obligatory_parents)),palim))
            
            if obligatory_parents:
                msg = "( with implicit parents {0} )".format(','.join(sorted(obligatory_parents)))
                def this_local_score(ch,pa):
                    # just always add in the obligatory parents
                    return local_score(ch,obligatory_parents_tuple + pa)
            else:
                this_local_score = local_score
                msg = ''
            
            score, ub = this_local_score(child,())
            scores_computed += 1

            child_dkt = {frozenset().union(obligatory_parents):score}   # output
            previous_layer = {():(score,ub)}  # frontier
            search_nodes_expanded += 1
            
            for pasize in range(1,thispalim+1):
                new_layer = {}
                for old_parentset in previous_layer:
                    last_idx = -1 if old_parentset == () else thisvaridx[old_parentset[-1]]
                    for new_parent in unfixed_parents[last_idx+1:]:

                        parents = old_parentset + (new_parent,)

                        ok = True
                        for pair in forbidden_pairs:
                            if pair.issubset(parents):
                                ok = False
                                break
                        if not ok:
                            continue
                        
                        # check that all subsets smaller by one exist in previous layer,
                        # and get best (i.e. highest) score and best (i.e. lowest) upper bound
                        bss = None
                        lub = None
                        for i in range(pasize):
                            try:
                                old_score, old_ub = previous_layer[parents[:i]+parents[i+1:]]
                            except KeyError:
                                bss = None
                                break
                            bss = old_score if bss is None else max(bss,old_score)
                            # An upper bound of None means no upper bound was computed
                            if old_ub is not None:
                                lub = old_ub if lub is None else min(lub,old_ub)


                        if bss is None or (lub is not None and bss >= lub):
                            # some subset is exponentially pruned, so don't score
                            # or: best we can hope for for parents (i.e. lub) is no better than some existing subset
                            # of parents (i.e. the one with score bss)
                            if verbose > 1 and (bss is not None and lub is not None and bss >= lub):
                                print('Pruning and not scoring {0}<-{2}{4}:\n\
                                \tFor child variable {0}, {1} is an upper bound for\n\
                                \t {2} and its supersets\n\tand some subset of {2} has score {3}'.format(child,lub,parents,bss,msg))
                            continue

                        # get local score and upper bound on any superset formed by adding
                        # BN variables coming after parents in bn_variables
                        score, ub = this_local_score(child,parents)
                        scores_computed += 1
                        
                        # if this parent set has a score exceeding that of any subsets then (and only then)
                        # is the score worth keeping
                        if score > bss:
                            child_dkt[frozenset(parents).union(obligatory_parents)] = score
                        
                        # the following line only needed to compensate for poor upper bounds
                        # since ub should already be as low as possible
                        if ub is not None and lub is not None:
                            ub = min(ub,lub)

                        best = max(score,bss)
                        #print ('check', best, ub)
                        if pasize < thispalim:
                            # not the last layer yet
                            if ub is not None and ub <= best:
                                # none of the proper supersets worth keeping
                                if verbose > 1:
                                    print('Pruning:\n \tFor child variable {0} {4} a subset of {1} has score {2} \n\
                                    \tand {3} is an upper bound on proper supersets of {1},'.format(child, parents, best,ub, msg))
                            else:
                                # expand frontier
                                new_layer[parents] = (best,ub)
                                search_nodes_expanded += 1
                previous_layer = new_layer
            score_dkt[child] = child_dkt
            if verbose > 1:
                print('{0} local scores stored for child variable {1}'.format(len(child_dkt),child))

        if verbose > 1:
            print('Search nodes expanded = {0}, scores computed = {1}, scores kept = {2}'.format(
                search_nodes_expanded,scores_computed,sum([len(d) for d in score_dkt.values()])))
        return score_dkt


    def _return_unpruned_local_scores(self,local_score,palim):
        score_dkt = {}
        bn_variables = self.bn_variables        
        bn_variables_set = frozenset(bn_variables)
        for child in bn_variables:

            forbidden_parents = frozenset([child]+[parent for (parent,x) in self.forbidden_arrows if x == child])
            obligatory_parents = frozenset([parent for (parent,x) in self.obligatory_arrows if x == child])
            forbidden_pairs = frozenset(
                [frozenset((x,y)) for (a,b,s) in self.obligatory_conditional_independences if child in s for x in a for y in b])
            pot_parents = bn_variables_set - forbidden_parents

            child_dkt = {}
            for pasize in range(palim+1):
                for parents in combinations(pot_parents,pasize):
                    ok = True
                    for pair in forbidden_pairs:
                        if pair.issubset(parents):
                            ok = False
                            break
                    if ok and obligatory_parents.issubset(parents):
                        child_dkt[frozenset(parents)] = local_score(child,parents)[0]
            score_dkt[child] = child_dkt

        return score_dkt
        
        
    def return_local_scores(self,local_score_fun,palim=3,pruning=True):
        """
        Return a dictionary for each child variable where the keys
        are the child variables and the values map parent sets to 
        local scores.

        Not all parent sets are included. If `palim` is not None, 
        then only those parent sets of cardinality at most `palim` can be included. 

        Also, when `pruning=True`, a parent set is only included if its local score
        exceeds that of all its proper subsets.

        `local_score_fun` should be a function that computes a local
        score. 

        Args:
            local_score_fun(fun/None): If not None a local score function such that `local_score_fun(child,parents)`
             computes `(score,ub)` where `score` is the desired local score for `child` having parentset `parents`
             and `ub` is either `None` or an upper bound on the local score for `child` with any proper superset of `parents`
            palim(int/None): If not None then the maximal size of a parent set
            pruning(bool): Whether not to include parent sets which cannot be optimal.

        Returns:
           dict: A dictionary `dkt` such that `dkt[child][parentset]` is the local score for `child` having parent set `parentset` (where `parentset` is a frozenset).

        """

        p = len(self.bn_variables)
        if palim is None:
            palim = p-1
        else:
            palim = min(palim,p-1)

        if pruning:
            skores =  self._return_pruned_local_scores(local_score_fun,palim)
        else:
            skores =  self._return_unpruned_local_scores(local_score_fun,palim)

        # now ensure that scores are available for all families in the 'extra' dags in self._starts (represented as dictionaries)
        for dag in self._starts:
            for child in dag.nodes:
                parents = frozenset(dag.predecessors(child))
                if parents not in skores[child]:
                    skores[child][parents] = local_score_fun(child,parents)[0]
                    
        return skores
    
    def write_local_scores(self,f):
        '''
        Write local scores to a file

        Args:
         f (str/file): If a string the name of the file to write to (where "-" leads to writing
          to standard output). Otherwise a file object.
        '''
        _write_local_scores(self.family_scores,f)
        
        
    def input_local_scores(self, local_scores):
        '''Read local scores from a dictionary.

        Once this method has been run, methods for adding MIP variables, such as
        :py:meth:`add_variables_family <gobnilp.Gobnilp.add_variables_family>` and 
        :py:meth:`add_basic_variables <gobnilp.Gobnilp.add_basic_variables>`, can be used.

        Args:
            local_scores (dict) : Dictionary containing local scores.
             ``local_scores[child][parentset]`` is the score for ``child`` having parents ``parentset``
             where ``parentset`` is a frozenset.
        '''
        self._family_scores = local_scores
        if not hasattr(self,'_bn_variables'):
            self._bn_variables = sorted(self._family_scores)

    def add_constraints_one_dag_per_MEC(self,dynamic=True,careful=False):
        '''Adds a constraint that only one DAG per Markov equivalence class is feasible.

        The constraint is effected by adding appropriate lazy constraints when Gurobi
        generates a DAG solution. 

        Args:
            dynamic (bool): If True then which DAG is feasible for each Markov equivalence
             class is arbitrary (it will be the first one Gurobi comes across). If false
             then the representative DAG is fixed 
             (and is determined by the method :py:meth:`dec2mag <gobnilp.Gobnilp.mec2dag>`).
            careful (bool): If True then all the lazy constraints are stored (not just posted) 
             to ensure that new solutions satisfy them. (The value for `careful` is ignored if
             `dynamic` is False.)
        '''
        self._one_dag_per_MEC = True
        self.Params.LazyConstraints = 1
        self._MEC_constraint_dynamic = dynamic
        if dynamic:
            self._MEC_constraint_careful = careful
            if careful:
                self._mecrep = {}
            
    def sol2fvs(self):
        '''Extracts the family variables set to true by some Gurobi solution
        
        The solution corresponding to the current value of Gurobi parameter ``SolutionNumber``
        is used.

        Returns:
            tuple: A pair of lists. The first list contains the families as ``(child,parents)``
            tuples, where ``parents`` is a frozenset. The second list contains Gurobi binary
            MIP variables for the families.
        '''
        families = []
        fvs = []
        family = self.family
        for v in self.bn_variables:
            for parent_set, var in list(family[v].items()):
                if var.Xn > 0.5:
                    families.append((v,parent_set))
                    fvs.append(var)
                    break
        return families, fvs


    def _enforce_MEC_representative(self):
        '''Enforces a constraint that the Markov equivalence class corresponding to the
        current solution has only one feasible DAG (its distinguished representative).

        This should only be called from a callback with ``where == GRB.Callback.MIPSOL``
        i.e. when we have a solution and where it is guaranteed that the solution is a DAG.

        This method should only be called if 
        :py:meth:`add_constraints_one_dag_per_MEC <gobnilp.Gobnilp.add_constraints_one_dag_per_MEC>`
        has previously been called to add the constraint. The enforcement method depends on how the ``dynamic``
        and ``careful`` flags were set when adding the constraint.

        If ``dynamic = True`` when the constraint was added then the DAG given by the current solution will normally become the 
        distinguished representative of its Markov equivalence class.
        In this case the current solution is not cut off.
        An exception to this occurs if (1) we are being careful 
        (we had ``careful=True`` when the constraint was added) and 
        (2) we have previously seen a different DAG in this Markov equivalence class - but somehow
        this did not prevent the current solution being proposed. In this case the earlier
        DAG is set (again) to be the distinguished representative.

        If ``dynamic = False``, the DAG defined by :py:meth:`dag2mec <gobnilp.Gobnilp.dag2mec>` is the 
        distinguished representative of the Markov equivalence class for 
        the current solution. In this case the current solution will be cut off unless it happens to be this
        distinguished representative.
        '''
        dag = []
        family_list = self.family_list
        for i, val in enumerate(self.cbGetSolution(family_list)):
            if val > 0.5:      # not '== 1' due to numerical problems!
                dag.append(i)
        assert len(dag) == self.n
        mec = self.dag2mec(dag)
        if self._MEC_constraint_dynamic == False:
            dag = None
        self._add_constraints_one_DAG_per_MEC(mec,dag)

    def _add_constraints_one_DAG_per_MEC(self,mec,dag=None):
        '''Add the constraint that the DAG `dag` is the only feasible
        DAG in the input Markov equivalence class `mec`.

        If ``dag==None`` then the distinguished representative for `mec`
        generated by the method :py:meth:`mec2dag <gobnilp.Gobnilp.mec2dag>` is used.

        This method should only be called from within a callback where we have a solution,
        i.e. where ``where == GRB.Callback.MIPSOL``.

        The constraint is a conjunction of linear constraints.
        
        Args:
            mec (frozenset): A Markov equivalence class represented by a characteristic imset which is
              itself (sparsely) represented by the set of all sets with value 1 in the characteristic imset.
              This method assumes that `mec` is a valid characteristic imset without any checking.
            dag (iterable/None): A DAG represented by a collection of 
             (indices for) 'families', where each family is a BN variable 
             together with its parents. Or ``None``.
        '''
        if dag is None:
            # representative DAG is fixed
            rep_dag = self.mec2dag(mec)
        elif self._MEC_constraint_careful:
            try:
                # following assignment only happens
                # if constraint for this MEC previously added
                # but somehow the supplied dag has not been ruled out
                rep_dag = self._mecrep[mec]
            except KeyError:
                # following assignment occurs if this is the first DAG
                # for this MEC which has been encountered
                rep_dag = dag
                self._mecrep[mec] = rep_dag
        else:
            # simply assume/hope that this is the first DAG seen for this MEC
            rep_dag = dag
            
        if self._c2 is None:
            self._add_cimset_info()
        rhs = len(self._c2) + len(self._c3) - 1
        # construct a linear expression which takes its maximum
        # value of len(self._c2) + len(self._c3) iff
        # the family variable values encode a DAG in the MEC
        # represented by mec
        lexpr = LinExpr()
        for ci, fvlist in list(self._c2.items()) + list(self._c3.items()):
            if ci in mec:
                lexpr.addTerms([1]*len(fvlist),fvlist)
            else:
                lexpr.addTerms([-1]*len(fvlist),fvlist)
                lexpr.addConstant(1)

        if rep_dag is None:
            # this mec not allowed since its distinguished rep
            # not available
            self.cbLazy(lexpr, GRB.LESS_EQUAL, rhs)
        else:
            family_list = self.family_list
            for i in rep_dag:
                self.cbLazy(lexpr, GRB.LESS_EQUAL, rhs + family_list[i])

    def all_to_nx(self):
        '''
        Constructs a list of ``BN`` objects corresponding to each solution
        in order
        '''
        nxs = []
        for i in range(self.Solcount):
            self.Params.SolutionNumber = i
            nxs.append(self.to_nx())
        nxs = tuple(nxs)
        self._nxs = nxs

    make_bns = all_to_nx
    
    def to_nx(self):
        '''
        Returns a ``BN`` object corresponding to a solution

        The solution corresponding to the current value of Gurobi parameter ``SolutionNumber``
        is used.

        Returns:
         BN: A BN structure (DAG) corresponding to a solution.

        '''
        families, fvs = self.sol2fvs()
        dag = BN(score=self.PoolObjVal)
        for i, (child,parent_set) in enumerate(families):
            dag.add_node(child,local_score=fvs[i].Obj)
            dag.add_edges_from([(parent,child) for parent in parent_set])
        return dag

    def write_dot(self,path="bn.dot"):
        '''Write a learned DAG to file in DOT format

        The solution corresponding to the current value of Gurobi parameter ``SolutionNumber``
        is used.

        Args:
            path(str): The output file name
        '''
        from networkx.drawing.nx_agraph import write_dot
        # ignore buggy StopIteration exception
        try:
            write_dot(self.to_nx(),path)
        except RuntimeError:
            pass
        
    def write_pdf(self,path="bn.pdf"):
        '''Create a PDF document displaying a learned DAG

        The solution corresponding to the current value of Gurobi parameter ``SolutionNumber``
        is used.

        Note that this method also creates a DOT file which
        is not deleted by this method.

        Args:
            path(str): The output file name
        '''
        import subprocess
        self.write_dot()
        subprocess.call(['dot', '-Tpdf', 'bn.dot', '-o', path])
        
    # def print_simple_output(self):
    #     '''
    #     Prints a simple representation of a learned DAG to standard output

    #     There is one line for each family which includes the local score for that family.

    #     The solution corresponding to the current value of Gurobi parameter ``SolutionNumber``
    #     is used.
    #     '''
    #     print('**********')
    #     print('Optimal BN has score', self.PoolObjVal)
    #     print('**********')
    #     families, fvs = self.sol2fvs()
    #     for i, (child,parent_set) in enumerate(families):
    #         print('{0}<-{1} {2}'.format(child, ','.join(sorted(parent_set)), fvs[i].Obj))
    #     print('**********')


    def _add_cimset_info(self):
        '''Create all c-imset components of size 2 and 3 which might possibly
        take value 1
        '''
        c2 = {}
        c3 = {}
        for child, parent_dict in list(self.family.items()):
            for parent_set, fv in list(parent_dict.items()):
                for parent in parent_set:
                    try:
                        c2[frozenset([child,parent])].append(fv)
                    except KeyError:
                        c2[frozenset([child,parent])] = [fv]
                for (pa1,pa2) in combinations(parent_set,2):
                    try:
                        c3[frozenset([child,pa1,pa2])].append(fv)
                    except KeyError:
                        c3[frozenset([child,pa1,pa2])] = [fv]
        self._c2 = c2
        self._c3 = c3
    
    def add_variables_family(self,branch_priority=0,best_first_branch_priority=False):
        '''Adds binary Gurobi MIP family variables to the model

        This method should only be called after data (or local scores) have been read
        in using, for example, a method such as 
        :py:meth:`input_discrete_data <gobnilp.Gobnilp.input_discrete_data>`

        Args:
            branch_priority (int): The Gurobi branching priority for the 
             family variables. This value is ignored if `best_first_branch_priority` is True.
            best_first_branch_priority (bool): If True then the branching priority for the 
             family variables for any given child are (totally) ordered according to local score,
             with the higher scoring families given higher branching priority than lower ones.
        '''
        
        family = {}
        fv_list = []
        child_list = []
        parentset_list = []
        get_idx = {}
        n = 0
        for child, parent_dict in list(self.family_scores.items()):
            family[child] = {}
            for parent_set, score in list(parent_dict.items()):
                fv = self.addVar(
                    obj = score,
                    vtype = GRB.BINARY,
                    name="{0}<-{{{1}}}".format(child,",".join(sorted(parent_set)))
                    )
                fv.BranchPriority = branch_priority
                family[child][parent_set] = fv
                fv_list.append(fv)
                child_list.append(child)
                parentset_list.append(parent_set)
                get_idx[child,parent_set] = n
                n += 1
            if best_first_branch_priority:
                for i, parent_set in enumerate(self.ordered_parentsets[child]):
                    family[child][parent_set].BranchPriority = -i
        if self._verbose:
            print('%d family variables declared' % n, file=sys.stderr)
        self._mipvars['family'] = family
        self._mipvars['family_list'] = fv_list
        self._idxdicts['child'] = child_list
        self._idxdicts['parents'] = parentset_list
        self._idxdicts['get_family_index'] = get_idx

    def _add_variables_local_scores(self,branch_priority=0):
        '''For each variable create a continuous variable which is the local score
        of the selected parent set for that variable
        '''
        local_score = {}
        n = 0
        for child, score_dkt in list(self.family_scores.items()):
            v = self.addVar(
                ub = max(score_dkt.values()),
                lb = min(score_dkt.values()),
                obj = 1,
                vtype=GRB.CONTINUOUS)
            v.BranchPriority = branch_priority
            local_score[child] = v
            n += 1
        if self._verbose:
            print('%d local score variables declared' % n, file=sys.stderr)
        self._local_score = local_score

        
    def add_variables_arrow(self,branch_priority=0):
        '''Adds binary Gurobi MIP arrow variables to the model

        The arrow variable corresponding to ``(pa,ch)`` is set to 1 iff
        there is an arrow from ``pa`` to ``ch`` in a learned BN.

        To connect these variables appropriately to family variables it is necessary to 
        call :py:meth:`add_constraints_arrow_family <gobnilp.Gobnilp.add_constraints_arrow_family>`.

        All these variables are given objective value 0. (This can be overridden using the ``Obj`` attribute
        of the variable.)

        Args:
            branch_priority (int): The Gurobi branching priority for the 
             arrow variables.
        '''
        arrow = {}
        n = 0
        for (v1, v2) in permutations(self.bn_variables,2):
            v = self.addVar(vtype=GRB.BINARY,name="{0}->{1}".format(v1,v2))
            v.BranchPriority = branch_priority
            arrow[v1,v2] = v
            n += 1
        if self._verbose:
            print('%d arrow variables declared' % n, file=sys.stderr)
        self._mipvars['arrow'] = arrow

    def add_variables_total_order(self,branch_priority=0):
        '''Adds binary Gurobi MIP total order variables to the model

        The total order variable corresponding to ``(v1,v2)`` is set to 1 iff
        ``v1 > v2`` in the total order associated with a learned BN. Parents always come
        before children in the total order.

        To connect these variables appropriately to arrow variables it is necessary to 
        call :py:meth:`add_constraints_arrow_total_order <gobnilp.Gobnilp.add_constraints_arrow_total_order>`.

        All these variables are given objective value 0. (This can be overridden using the ``Obj`` attribute
        of the variable.)

        Args:
            branch_priority (int): The Gurobi branching priority for the 
             total order variables.
        '''
        total_order = {}
        n = 0
        for (v1, v2) in permutations(self.bn_variables,2):
            v = self.addVar(vtype=GRB.BINARY,name="{0}>{1}".format(v1,v2))
            v.BranchPriority = branch_priority
            total_order[v1,v2] = v
            n += 1
        if self._verbose:
            print('%d total_order variables declared' % n, file=sys.stderr)
        self._mipvars['total_order'] = total_order

        
    def add_variables_adjacency(self,branch_priority=0):
        '''Adds binary Gurobi MIP adjacency variables to the model

        The adjacency variable corresponding to ``{v1,v2}`` is set to 1 iff
        there is an arrow from ``v1`` to ``v2`` or an arrow from ``v2`` to ``v1``.

        To connect these variables appropriately to arrow variables it is necessary to 
        call :py:meth:`add_constraints_arrow_adjacency <gobnilp.Gobnilp.add_constraints_arrow_adjacency>`.

        All these variables are given objective value 0. (This can be overridden using the ``Obj`` attribute
        of the variable.)

        Args:
            branch_priority (int): The Gurobi branching priority for the 
             adjacency variables.

        '''
        adj = {}
        n = 0
        for (v1, v2) in combinations(self.bn_variables,2):
            v = self.addVar(vtype=GRB.BINARY,name="{0}-{1}".format(*sorted([v1,v2])))
            v.BranchPriority = branch_priority
            adj[frozenset([v1,v2])] = v
            n += 1
        if self._verbose:
            print('%d adj variables declared' % n, file=sys.stderr)
        self._mipvars['adjacency'] = adj


    def add_variables_genindex(self,branch_priority=0,earlyfirst=True):
        genindex = {}
        n = 0
        for bn_variable in self.bn_variables:
            for pos in range(self.n):
                v = self.addVar(vtype=GRB.BINARY)
                genindex[bn_variable,pos] = v
                if earlyfirst:
                    v.BranchPriority = self.n-pos
                else:
                    v.BranchPriority = branch_priority
                n += 1
        if self._verbose:
            print('%d generation index variables declared' % n, file=sys.stderr)
        self._mipvars['generation_index'] = genindex

        
    def add_variables_gen(self,branch_priority=0):
        '''Adds generation variables to the model

        A generation number for a variable in a DAG is an integer such that any variable 
        has a generation number stricty greater than any of its parents.

        To connect these variables appropriately to arrow variables it is necessary to 
        call :py:meth:`add_constraints_gen_arrow_indicator <gobnilp.Gobnilp.add_constraints_gen_arrow_indicator>`.

        To set the sum of all generation numbers to n*(n-1)/2 use 
        :py:meth:`add_constraints_sumgen <gobnilp.Gobnilp.add_constraints_sumgen>`.

        All these variables are given objective value 0. (This can be overridden using the ``Obj`` attribute
        of the variable.)

        Args:
            branch_priority (int): The Gurobi branching priority for the 
             generation variables.
        '''
        gen = {}
        n = 0
        max_num_before = self.n-1
        for bn_variable in self.bn_variables:
            v = self.addVar(
                ub = max_num_before,
                lb = 0,
                vtype=GRB.INTEGER
            )
            gen[bn_variable] = v
            v.BranchPriority = branch_priority
            n += 1
        if self._verbose:
            print('%d generation variables declared' % n, file=sys.stderr)
        self._mipvars['generation'] = gen

    def add_variables_gendiff(self,branch_priority=0):
        '''Adds variables representing the difference in generation number
        between distinct BN variables

        Generation and generation difference variables are connected appropriately with 
        :py:meth:`add_constraints_gendiff <gobnilp.Gobnilp.add_constraints_gendiff>`.

        All these variables are given objective value 0. (This can be overridden using the ``Obj`` attribute
        of the variable.)

        Args:
            branch_priority (int): The Gurobi branching priority for the 
             generation difference variables.
        '''
        gendiff = {}
        max_num_before = self.n-1
        n = 0
        bn_variables = self.bn_variables
        for i, v1 in enumerate(bn_variables):
            for v2 in bn_variables[i+1:]:
                v = self.addVar(
                    ub = max_num_before,
                    lb = -max_num_before,
                    vtype=GRB.INTEGER)
                v.BranchPriority = branch_priority
                gendiff[v1,v2] = v
                n += 1
        if self._verbose:
            print('%d generation difference variables declared' % n, file=sys.stderr)
        self._mipvars['generation_difference'] = gendiff

        
    def add_variables_absgendiff(self,branch_priority=0):
        '''Adds variables representing the absolute difference in generation number
        between each pair of distinct BN variables.

        These variables are constrained to have a **lower bound of 1**. So as long as
        constraints are posted connecting these variables to the *generation_difference* variables
        and thus ultimately to the *generation* variables, then each BN variable will have
        a different generation number.

        Calling :py:meth:`add_constraints_absgendiff <gobnilp.Gobnilp.add_constraints_absgendiff>` ensure that
        these variables indeed are equal to the absolute values of generation difference variables.

        Generation variables are added with :py:meth:`add_variables_gen <gobnilp.Gobnilp.add_variables_gen>`.
        Generation difference variables are added with :py:meth:`add_variables_gendiff <gobnilp.Gobnilp.add_variables_gendiff>`.
        See the documentation for these two methods for details of how to add appropriate constraints.

        All these variables are given objective value 0. (This can be overridden using the ``Obj`` attribute
        of the variable.)

        Args:
            branch_priority (int): The Gurobi branching priority for the 
             absolute generation difference variables.
        '''
        absgendiff = {}
        max_num_before = self.n-1
        n = 0
        for (v1, v2) in combinations(self.bn_variables,2):
            v = self.addVar(
                ub = max_num_before,
                lb = 1,
                vtype=GRB.INTEGER)
            v.BranchPriority = branch_priority
            absgendiff[frozenset([v1,v2])] = v
            n += 1
        if self._verbose:
            print('%d absolute gen difference variables declared' % n, file=sys.stderr)
        self._mipvars['absolute_generation_difference'] = absgendiff
                
        
    def add_basic_variables(self):
        '''Adds the most useful Gurobi MIP variables

        Adds the variables added by the following methods:

        * :py:meth:`add_variables_family <gobnilp.Gobnilp.add_variables_family>`
        * :py:meth:`add_variables_arrow <gobnilp.Gobnilp.add_variables_arrow>`
        * :py:meth:`add_variables_adjacency <gobnilp.Gobnilp.add_variables_adjacency>`

        Arrow and adjacency variables are given a higher branching priority than family
        variables to encourage a balanced search tree.
        '''
        self.add_variables_family()
        #self.add_variables_family(best_first_branch_priority=True)
        self.add_variables_arrow(branch_priority=10)
        self.add_variables_adjacency(branch_priority=10)
        #self.add_variables_arrow()
        #self.add_variables_adjacency()
        #self.add_variables_total_order()
        #self.add_variables_gen()
        #self.add_variables_gendiff()
        #self.add_variables_absgendiff()

    def add_constraints_gen_index_link(self):
        '''Adds the constraint linking generation to generation index variables
        '''
        n = 0
        genindex = self.generation_index
        generation = self.generation
        for bn_variable in self.bn_variables:
            self.addConstr(LinExpr(range(self.n),[genindex[bn_variable,pos] for pos in range(self.n)]),
                           GRB.EQUAL,
                           generation[bn_variable])
            n += 1
        if self._verbose:
            print('%d constraints linking generation to generation index variables posted' % n, file=sys.stderr)


    def add_constraints_4b(self):
        '''Adds "4B" constraints. See
        `Bayesian Network Structure Learning with Integer Programming:
        Polytopes, Facets and Complexity (Cussens et al, JAIR) <https://jair.org/index.php/jair/article/view/11041/26213>`_
        for details
        '''
        n = 0
        bnvarset = frozenset(self.bn_variables)
        family = self.family
        for (a,b) in combinations(bnvarset,2):
            abset = frozenset((a,b))
            for (c,d) in combinations(bnvarset.difference((a,b)),2):
                vs = []
                childs = set()
                cdset = frozenset((c,d))
                for (x,y) in [(a,b),(b,a)]:
                    for parentset, fv in family[x].items():
                        if y in parentset or cdset <= parentset:
                            childs.add(x)
                            vs.append(fv)
                for (x,y) in [(c,d),(d,c)]:
                    for parentset, fv in family[x].items():
                        if y in parentset and not parentset.isdisjoint(abset):
                            childs.add(x)
                            vs.append(fv)
                if len(childs) > 2 and len(vs) > 2:
                    self.addConstr(LinExpr([1.0] * len(vs),vs) <= 2)
                    n += 1
        if self._verbose:
            print('%d 4B constraints posted' % n, file=sys.stderr)  
                
            
    def add_constraints_genindex(self):
        '''Adds the constraint that each variable has exactly one
        generation index and that each of these indices is distinct
        '''
        n = 0
        genindex = self.generation_index
        for bn_variable in self.bn_variables:
            self.addConstr(LinExpr([1.0] * self.n,[genindex[bn_variable,pos] for pos in range(self.n)]),
                           GRB.EQUAL,
                           1)
            n += 1
        for pos in range(self.n):
            self.addConstr(LinExpr([1.0] * self.n,[genindex[bn_variable,pos] for bn_variable in self.bn_variables]),
                           GRB.EQUAL,
                           1)
            n += 1
        if self._verbose:
            print('%d gen index constraints posted' % n, file=sys.stderr)

        
    def add_constraints_oneparentset(self):
        '''Adds the constraint that each child has exactly one parent set.
        '''
        n = 0
        for parentset_dict in list(self.family.values()):
            self.addConstr(LinExpr([1.0] * len(parentset_dict),list(parentset_dict.values())),
                           GRB.EQUAL,
                           1)
            n += 1
        if self._verbose:
            print('%d constraints insisting on exactly one parent set for each variable' % n, file=sys.stderr)

    def _extend(self,ss):
        '''private method used by add_constraints_setpacking
        '''
        family = self.family
        new_ss = []
        for si, s in ss:
            # s is a tuple of variable indices
            for i in range(si[-1]+1,self.n):
                ok1 = True
                new_elt = self.bn_variables[i]
                new_s = s | frozenset([new_elt])
                for child in new_s:
                    others = new_s - frozenset([child])
                    ok2 = False
                    for parentset in family[child]:
                        if others <= parentset:
                            ok2 = True
                            break
                    if not ok2:
                        ok1 = False
                        break
                if ok1:
                    new_ss.append((si+(i,),new_s))
        return tuple(new_ss)

    def _spc(self,ss):
        '''private method used by add_constraints_setpacking
        '''
        family = self.family
        for x, s in ss:
            fvs = []
            #nonfvs = []
            for child in s:
                others = s - frozenset([child])
                for parentset, fv in list(family[child].items()):
                    if others <= parentset:
                        fvs.append(fv)
                    #else:
                    #    nonfvs.append(fv)
            #v = self.addVar(vtype=GRB.BINARY)
            #if len(s) == 3:
            #    v.BranchPriority = 10
            #if len(s) == 4:
            #    v.BranchPriority = 10
            #self.addConstr(LinExpr([1]*len(fvs),fvs),GRB.EQUAL,v)
            self.addConstr(LinExpr([1]*len(fvs),fvs),GRB.LESS_EQUAL,1)
            #self.addConstr(LinExpr([1]*len(nonfvs),nonfvs),GRB.GREATER_EQUAL,len(s)-1)
        
    
    def add_constraints_setpacking(self):
        '''Adds constraints like a<-b,c + b<-a,c + c<-a,b <= 1.
        That is an example for a "triple". Also adds similar constraints for 4-tuples.
        '''
        singles = []
        for i, v in enumerate(self.bn_variables):
            singles.append(((i,),frozenset([v])))
        pairs = self._extend(singles)
        triples = self._extend(pairs)
        quads = self._extend(triples)
        self._spc(triples)
        self._spc(quads)
        if self._verbose:
            print('%d set packing constraints declared' % (len(triples) + len(quads)), file=sys.stderr)

    def add_constraints_arrow_family(self):
        '''Adds constraints linking arrow variables to family variables

        If ``pa`` is not a parent of ``ch`` in any family then the corresponding
        arrow variable ``ch<-pa`` is simply removed.
        '''
        n = 0
        m = 0
        arrow = self.arrow
        family = self.family
        for (pa, ch), arrow_var in list(arrow.items()):
            vs = [arrow_var]
            non_vs = [arrow_var]
            vals = [-1]
            non_vals = [1]
            for parentset, fv in list(family[ch].items()):
                if pa in parentset:
                    vs.append(fv)
                    vals.append(1)
                else:
                    non_vs.append(fv)
                    non_vals.append(1)
            if len(vs) == 1:
                # arrow can never occur
                self.remove(arrow_var)
                del arrow[pa,ch]
                m += 1
            else:
                self.addConstr(LinExpr(vals,vs),GRB.EQUAL,0)
                self.addConstr(LinExpr(non_vals,non_vs),GRB.EQUAL,1)
                n += 2
        if self._verbose:
            print('%d constraints linking arrows to family variables declared' % n, file=sys.stderr)
            print('%d arrow variables removed' % m, file=sys.stderr)

    def add_constraints_arrow_total_order(self):
        '''Adds constraints linking arrow variables to total order variables
        '''
        n = 0
        total_order = self.total_order
        for (pa, ch), arrow_var in list(self.arrow.items()):
            self.addConstr(arrow_var <= total_order[ch,pa]) # ch > pa in order
            n += 1
        if self._verbose:
            print('%d constraints linking arrows to total order variables declared' % n, file=sys.stderr)

    def add_constraints_total_order(self,lazy=0):
        '''Adds constraints so that total order variables
        represent a total order

        Args:
         lazy(int): Controls the 'laziness' of these constraints by settng the
          Lazy attribute of the constraints.         
          See `the Gurobi documentation <https://www.gurobi.com/documentation/8.1/refman/lazy.html>`_
        '''
        n = 0
        total_order = self.total_order
        bn_variables = self.bn_variables
        for (v1, v2) in combinations(bn_variables,2):
            self.addConstr(total_order[v1,v2] + total_order[v2,v1] == 1)
            n += 1
        for v1, v2 in permutations(bn_variables,2):
            for v3 in bn_variables:
                if v3 != v1 and v3 != v2:
                    constr = self.addConstr(total_order[v1,v2] + total_order[v2,v3] + total_order[v3,v1] <= 2)
                    constr.Lazy = lazy
                    n += 1
        if self._verbose:
            print('%d total order constraints declared' % n, file=sys.stderr)


            
    def add_constraints_arrow_adjacency(self):
        '''Add constraints that there is an adjacency between
        v1 and v2 in the undirected skeleton if there is either an 
        arrow from v1 to v2, or an arrow from v2 to v1 in the DAG
        '''
        n = 0
        m = 0
        arrow = self.arrow
        adjacency = self.adjacency
        for pair, adj_var in list(adjacency.items()):
            [v1, v2] = list(pair)
            coeffs = [-1]
            vs = [adj_var]
            for e in [(v1,v2),(v2,v1)]:
                if e in arrow:
                    coeffs.append(1)
                    vs.append(arrow[e])
            if len(vs) > 1:
                self.addConstr(
                    LinExpr(coeffs,vs),GRB.EQUAL,0)
                n += 1
            else:
                self.remove(adj_var)
                del adjacency[pair]
                m += 1
        if self._verbose:
            print('%d constraints linking arrows to adjs declared' % n, file=sys.stderr)
            print('%d adjacency variables removed' % m, file=sys.stderr)

    def add_constraints_sumgen(self):
        '''Adds the constraint that sum of gen numbers is n*(n-1)/2
        '''
        n = self.n
        self.addConstr(
            LinExpr([1.0]*n,list(self.generation.values())),
            GRB.EQUAL,
            n*(n-1)/2
        )
        if self._verbose:
            print('1 constraint that the sum of gen numbers is',  n*(n-1)/2, file=sys.stderr)

    def add_constraints_gen_arrow_indicator(self):
        '''Adds constraints stating that an arrow from parent to child
        means that the child's generation number is strictly greater than the parent's.
        '''
        n = 0
        generation = self.generation
        for (pa,ch), arrow_var in list(self.arrow.items()):
            self.addConstr((arrow_var == 1) >> (generation[ch] <= generation[pa] - 1))
            n += 1
        if self._verbose:
            print('%d indicator constraints linking arrows to gen numbers  declared' % n, file=sys.stderr)

    def add_constraints_gendiff(self):
        '''Adds constraints linking generation difference variables to generation variables.
        '''
        n = 0
        generation = self.generation
        for (v1,v2), gendiffvar in list(self.generation_difference.items()):
            self.addConstr(
                LinExpr([1,-1,-1],[generation[v1],generation[v2],gendiffvar]),
                GRB.EQUAL,
                0)
            n += 1
        if self._verbose:
            print('%d constraints linking gen diff to gen declared' % n, file=sys.stderr)

            
    def add_constraints_absgendiff(self):
        '''Adds constraints linking generation difference variables to 
           absolute generation difference variables
        '''
        n = 0
        absolute_generation_difference = self.absolute_generation_difference
        for (v1,v2), gendiffvar in list(self.generation_difference.items()):
            self.addConstr(absolute_generation_difference[frozenset([v1,v2])] == abs_(gendiffvar))
            n += 1
        if self._verbose:
            print('%d indicator constraints linking abs gen diff to gen diff variables  declared' % n, file=sys.stderr)

    def add_constraints_chordal(self):
        '''Adds simple constraints to rule out non-chordal DAGs
        i.e. those without v-structures (aka immoralities)

        Constraints are roughly of this sort:
         a<-{b,c} + b<-{a,c} + c<-{a,b} <= a-b 
        '''
        n = 0
        dkt = {}
        for child, parent_dict in list(self.family.items()):
            for parent_set, fv in list(parent_dict.items()):
                for pa1, pa2 in combinations(parent_set,2):
                    ci = frozenset([child,pa1,pa2])
                    try:
                        dkt[ci].append(fv)
                    except KeyError:
                        dkt[ci] = [fv]
        adjacent = self.adjacent
        for ci, fvlist in list(dkt.items()):
            for pair in combinations(ci,2):
                self.addConstr(LinExpr([1]*len(fvlist),fvlist), GRB.LESS_EQUAL, adjacent[frozenset(pair)])
                n += 1
        if self._verbose:
            print('%d constraints ruling out immoralities declared' % n, file=sys.stderr)

    def add_constraints_clusters(self,cluster_cuts=True,matroid_cuts=False,matroid_constraints=False):
        '''Adds cluster constraints

        For any cluster of BN variable, the cluster constraint states that at least one element
        in the cluster has no parents in that cluster.

        These constraints are always added lazily, since there are exponentially many of them.

        Args:
            cluster_cuts(bool): If True then cluster constraints are added as cuts (i.e. as soon as the linear
              relaxation is solved). If False, we wait until a candidate integer solution is found.
            matroid_cuts(bool): If True then cuts corresponding to rank 2 matroids are also added.
            matroid_constraints(bool): If True then constraints corresponding to rank 2 matroids are added
              when an integer solution corresponding to a cyclic digraph is generated.        
        '''
        self._enforcing_cluster_constraints = True
        self._enforcing_matroid_constraints = matroid_constraints
        self._adding_cluster_cuts = cluster_cuts
        self._adding_matroid_cuts = matroid_cuts
        self.Params.LazyConstraints = 1
        if self._verbose:
            print('(Lazy) "cluster" constraints in use', file=sys.stderr)

    def add_constraints_bests(self):
        best = {}
        for v, pas in self.ordered_parentsets.items():
            best[v] = self.family[v][pas[0]]
        for x in self._best_subsets:
            constr = self.addConstr(LinExpr([1]*len(x),[best[y] for y in x]) >= 1) 
            #constr.Lazy = 2
        
    def add_basic_constraints(self):
        '''Adds the most useful constraints

        Adds the constraints added by the following methods:

        * :py:meth:`add_constraints_oneparentset <gobnilp.Gobnilp.add_constraints_oneparentset>`
        * :py:meth:`add_constraints_setpacking <gobnilp.Gobnilp.add_constraints_setpacking>`
        * :py:meth:`add_constraints_arrow_family <gobnilp.Gobnilp.add_constraints_arrow_family>`
        * :py:meth:`add_constraints_arrow_adjacency <gobnilp.Gobnilp.add_constraints_arrow_adjacency>`
        * :py:meth:`add_constraints_clusters <gobnilp.Gobnilp.add_constraints_clusters>`
        '''
        self.add_constraints_oneparentset()
        self.add_constraints_setpacking()
        self.add_constraints_arrow_family()
        self.add_constraints_arrow_adjacency()
        #self.add_constraints_arrow_total_order()
        #self.add_constraints_total_order()
        self.add_constraints_clusters()
        #self.add_constraints_sumgen()
        #self.add_constraints_gen_arrow_indicator()
        #self.add_constraints_gendiff()
        #self.add_constraints_absgendiff()
        #self.add_constraints_4b()
        #self.add_constraints_bests()
        
    def _mycallback(self,where):
        '''callback for adding cuts and lazy constraints
        '''
        max_cluster_size = self._max_cluster_size
        if (where == GRB.Callback.MIPNODE and
            self.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL and
            self.cbGet(GRB.Callback.MIPNODE_NODCNT) % 20 == 0): 

            # optionally don't look for cuts if this has already been done
            # 'too' often
            if (self._user_cuts_rounds_limit is not None and
                self._user_cuts_rounds_count > self._user_cuts_rounds_limit):
                return

            # optionally don't look for cuts if the current bound is not sufficiently
            # distant from the bound we had last time we looked for cuts
            # this is to prevent stalling
            if self._user_cuts_stalling_limit is not None:
                current_bound = self.cbGet(GRB.Callback.MIPNODE_OBJBND)
                if self._last_bound - current_bound < self._user_cuts_stalling_limit:
                    return
                self._last_bound = current_bound
            
            if self._adding_cluster_cuts:
                self._subip(cutting = True,max_cluster_size=max_cluster_size)
                self._user_cuts_rounds_count += 1
            if self._adding_matroid_cuts:
                self._matroid_subip(cutting = True)
        elif where == GRB.Callback.MIPSOL:

            # optionally don't look for constraints if this has already been done
            # 'too' often - used for solving relaxed versions of the problem
            if (self._user_enforcement_rounds_limit is not None and
                self._user_enforcement_rounds_count > self._user_enforcement_rounds_limit):
                return

            is_a_dag = True
            if self._enforcing_cluster_constraints:
                #print(self.cbGet(GRB.Callback.MIPSOL_NODCNT))
                is_a_dag = not self._subip(cutting = False,max_cluster_size=max_cluster_size)
                self._user_enforcement_rounds_count += 1
            if is_a_dag:
                self._enforce_lazy_user_constraints()
            if not is_a_dag and self._enforcing_matroid_constraints:
                self._matroid_subip(cutting = False)
            if is_a_dag and self._one_dag_per_MEC == True:
                self._enforce_MEC_representative()

    def _enforce_lazy_user_constraints(self):
        # first get representation of the DAG as a gobnilp.BN object
        digraph = BN()
        for child, dkt in self.family.items():
            digraph.add_node(child)
            for parentset, v in dkt.items():
                if self.cbGetSolution(v) > 0.5:
                    for pa in parentset:
                        digraph.add_edge(pa,child)
                    break
        self._lazy_user_constraints(digraph)

    def _lazy_user_constraints(self,digraph):
        '''
        forbidden and obligatory ancestors the only lazy constraints at present
        '''
        for (an,de) in self.forbidden_ancestors:
            try:
                path = nx.shortest_path(digraph,an,de)
                num_arrows = len(path)-1
                # e.g. a path from an->v1->v2->de is [an,v1,v2,de]
                # just rule out this specific pathx4
                self.cbLazy(LinExpr([1]*num_arrows,
                                    [self.arrow[path[i],path[i+1]] for i in range(num_arrows)]), GRB.LESS_EQUAL, num_arrows-1)
                # just add first constraint
                return
            except nx.NetworkXNoPath:
                pass
            
        for (an,de) in self.obligatory_ancestors:
            ans = nx.ancestors(digraph,de)
            if an not in ans:
                ans.add(de)
                fvs = [self.family[v][frozenset(digraph.predecessors(v))] for v in ans]
                # this ancestral graph is no good ..
                self.cbLazy(LinExpr([1]*len(fvs),fvs), GRB.LESS_EQUAL, len(fvs)-1)
                # just add first constraint
                return

                
        for (a,b,s) in self.obligatory_conditional_independences:
            ok, mag_nodes = digraph.satisfy_ci(a,b,s)
            if not ok:
                if self._verbose > 1:
                    print('Rejecting:\n{0}\nsince must have {1}\n'.format(
                        '\n'.join('{0}<-{1}'.format(v,','.join(digraph.predecessors(v))) for v in mag_nodes),
                        '{{{0}}} _|_ {{{1}}} | {{{2}}}'.format(','.join(a),','.join(b),','.join(s))))
                fvs = [self.family[v][frozenset(digraph.predecessors(v))] for v in mag_nodes]
                # this ancestral graph is no good ..
                self.cbLazy(LinExpr([1]*len(fvs),fvs), GRB.LESS_EQUAL, len(fvs)-1)
                # just add first constraint
                return


    def _best_subip(self):
        best_subip = Model("best subip")
        best_subip.Params.OutputFlag = 0
        best_subip.Params.PoolSearchMode = 2
        best_subip.Params.PoolSolutions = 50
        best = {}
        ordered_parentsets = self.ordered_parentsets 
        for v in self.bn_variables:
            if len(ordered_parentsets[v]) > 1:
                best[v] = best_subip.addVar(vtype=GRB.BINARY,obj=1)
        best_subip.addConstr(LinExpr([1.0]*len(best),list(best.values())) >= 1)
        for v in self.bn_variables:
            for bestpa in ordered_parentsets[v][0]:
                best_subip.addConstr(best[v] <= best[bestpa])
        best_subip.optimize()
        best_subsets = []
        for i in range(best_subip.Solcount):
            best_subip.Params.SolutionNumber = i
            bs = set()
            for v, vv in best.items():
                if vv.Xn > 0.5:
                    bs.add(v)
                    #print(v, ordered_parentsets[v][0])
            for x in best_subsets:
                if bs.issubset(x):
                    break
            else:
                best_subsets.append(bs)
        self._best_subsets = best_subsets


                        
                
    def _matroid_subip(self,cutting):
        '''experimental: generate rank 2 matroid cuts but too slow
        and cuts do not seem too useful
        '''
        matroid_subip = Model("matroid subip")
        matroid_subip.Params.OutputFlag = 0
        matroid_subip.ModelSense = -1
        matroid_subip.Params.Cutoff = -1.5
        matroid_subip.Params.PoolSolutions = 20
        matroid_subip.Params.PoolSearchMode = 2
        matroid_subip.Params.TimeLimit = 5
        matroid_subip.Params.MIPFocus = 1

        
        # Dictionaries for matroid subIP variables
        y = {}           # ground set indicators
        sub_fvs = {}     # indicates whether a family variable is in the cut
        circuits2 = {}   # indicates whether a subset of size 2 is a circuit
        not_circuits2 = {}   # indicates whether a subset of size 2 is not a circuit

        bn_variables = self.bn_variables
        
        try:
            bn_variables_pairs = self.bn_variables_pairs
            bn_variables_pairs_set = self._bn_variables_pairs_set
            bn_variables_triples_of_pairs = self._bn_variables_triples_of_pairs             
        except AttributeError:
            bn_variables_pairs = tuple([x for x in combinations(bn_variables,2)])
            bn_variables_pairs_set = tuple([frozenset(x) for x in bn_variables_pairs])
            bn_variables_triples_of_pairs = []
            for (v1,v2,v3) in combinations(bn_variables,3):
                bn_variables_triples_of_pairs.append((frozenset([v1,v2]),frozenset([v1,v3]),frozenset([v2,v3])))
            bn_variables_triples_of_pairs = tuple(bn_variables_triples_of_pairs)
            self._bn_variables_pairs = bn_variables_pairs
            self._bn_variables_pairs_set = bn_variables_pairs_set
            self._bn_variables_triples_of_pairs = bn_variables_triples_of_pairs             

        for v in bn_variables:
            y[v] = matroid_subip.addVar(vtype=GRB.BINARY,obj=-1)

        for pairset in self._bn_variables_pairs_set:
            circuits2[pairset] = matroid_subip.addVar(vtype=GRB.BINARY)
            not_circuits2[pairset] = matroid_subip.addVar(vtype=GRB.BINARY)

        if cutting:
            fv_vals = self.cbGetNodeRel(self.family_list)
        else:
            fv_vals = self.cbGetSolution(self.family_list)

        child_list = self.child
        parentset_list = self.parents
        for i, relval in enumerate(fv_vals):
            if relval > 0:
                child = child_list[i]
                parentset = parentset_list[i]
                #print child, '<-', ','.join(sorted(parentset)), relval
                v = matroid_subip.addVar(vtype=GRB.BINARY,obj=relval)
                try:
                    sub_fvs[child][parentset] = v
                except KeyError:
                    sub_fvs[child] = {parentset:v}

        matroid_subip.update()

        matroid_subip.addConstr(LinExpr([1]*len(y),list(y.values())), GRB.GREATER_EQUAL, 4)
        #matroid_subip.addConstr(LinExpr([1]*len(y),y.values()), GRB.LESS_EQUAL, 6)
        
        for i, (v1,v2) in enumerate(self._bn_variables_pairs):
            pairset = self._bn_variables_pairs_set[i]
            matroid_subip.addConstr(circuits2[pairset] + not_circuits2[pairset] <= y[v1])
            matroid_subip.addConstr(circuits2[pairset] + not_circuits2[pairset] <= y[v2])
            matroid_subip.addConstr(y[v1] + y[v2] <= 1 + circuits2[pairset] + not_circuits2[pairset])

        for pairset1, pairset2, pairset3 in self._bn_variables_triples_of_pairs:
            matroid_subip.addConstr(circuits2[pairset1] + circuits2[pairset2] + not_circuits2[pairset3] <= 2)
            matroid_subip.addConstr(circuits2[pairset1] + circuits2[pairset3] + not_circuits2[pairset2] <= 2)
            matroid_subip.addConstr(circuits2[pairset2] + circuits2[pairset3] + not_circuits2[pairset1] <= 2)

        # want at least one circuit, call it {a,b,c} of size 3, so none of {a,b}, {a,c} and {b,c} can be circuits
        # the following constraint is not sufficient to ensure the existence of a circuit of size 3
        matroid_subip.addConstr(3, GRB.LESS_EQUAL, LinExpr([1]*len(not_circuits2),list(not_circuits2.values())))

        for child, parent_dict in list(sub_fvs.items()):
            for parent_set, fv in list(parent_dict.items()):
                matroid_subip.addConstr(fv, GRB.LESS_EQUAL, y[child])
                n = len(parent_set)
                matroid_subip.addConstr(
                    fv, GRB.LESS_EQUAL, 
                    LinExpr([1]*(n + n*(n-1)//2),
                            [circuits2[frozenset([child,parent])] for parent in parent_set] +
                            [not_circuits2[frozenset([pa1,pa2])] for (pa1,pa2) in combinations(parent_set,2)])
                )

        matroid_subip.optimize()
                
        # if no matroid constraint found ...
        if matroid_subip.Status == GRB.CUTOFF:
            #print 'no matroid cuts found'
            return False

        for i in range(matroid_subip.Solcount):
            matroid_subip.Params.SolutionNumber = i

            ground_list = [yi for (yi,yv) in list(y.items()) if yv.Xn > 0.5]
            size2_circuits = [circuit for (circuit, v) in list(circuits2.items()) if v.Xn > 0.5]
            size3_circuits = []
            for triple in combinations(ground_list,3):
                triple = frozenset(triple)
                for c in size2_circuits:
                    if c <= triple:
                        break
                else:
                    size3_circuits.append(triple)

            # if ended up without any circuits of size 3
            # then no good
            if size3_circuits == []:
                continue

            size23_circuits = size2_circuits + size3_circuits
            
            # check that the matroid is connected
            connected = True
            for (v1,v2) in combinations(ground_list,2):
                pair = frozenset((v1,v2))
                for circuit in size23_circuits:
                    if pair <= circuit:
                        break
                else:
                    connected = False
                    break

            if not connected:
                continue
            
            #print
            #print 'Ground set', ground_list
            #print 'Circuits 2', size2_circuits
            #print 'Circuits 3', size3_circuits
            #print matroid_subip.objVal

            rhs = len(ground_list)-2
            lexpr = LinExpr()
            lhs_string = ''
            activity = 0.0
            for child in ground_list:
                intersecting_fvs = []
                non_intersecting_fvs = []
                for parent_set, fv in list(self.family[child].items()):
                    tmp = parent_set | frozenset([child])
                    for circuit in size23_circuits:
                        if child in circuit and circuit <= tmp:
                            intersecting_fvs.append(fv)
                            lhs_string += ' + {0}<-{1}'.format(child,','.join(sorted(parent_set)))
                            try:
                                activity += sub_fvs[child][parent_set].obj
                            except KeyError:
                                pass
                            break
                    else:
                        non_intersecting_fvs.append(fv)
                len_i = len(intersecting_fvs)
                len_ni = len(non_intersecting_fvs)
                if len_i > len_ni:
                    lexpr.addTerms([-1]*len_ni,non_intersecting_fvs)
                    rhs -= 1
                else:
                    lexpr.addTerms([1]*len_i,intersecting_fvs)
            if cutting:
                self.cbCut(lexpr, GRB.LESS_EQUAL, rhs)
            else:
                self.cbLazy(lexpr, GRB.LESS_EQUAL, rhs)
            #print lhs_string, '<=', len(ground_list)-2, 'activity = ', activity
        return True

    def _vanilla_cons(self,cutting,cluster):
        '''in each cluster one variable must come last
        and thus have all other cluster members available as
        parents
        '''
        lexpr = LinExpr()
        rhs = 1
        ordered_parentsets = self.ordered_parentsets
        family = self.family
        for child in cluster:
            child_dkt = family[child]
            n = len(ordered_parentsets[child])
            for i, parent_set in enumerate(ordered_parentsets[child]):
                if parent_set <= cluster:
                    if i < n / 2:
                        lexpr.addTerms([1]*(i+1),[child_dkt[ps] for ps in ordered_parentsets[child][:i+1]])
                    else:
                        lexpr.addTerms([-1]*(n-(i+1)),[child_dkt[ps] for ps in ordered_parentsets[child][i+1:]])
                        rhs -= 1
                    # don't consider parent sets worse than this one
                    break
        if cutting:
            self.cbCut(lexpr, GRB.GREATER_EQUAL, rhs)
        else:
            self.cbLazy(lexpr, GRB.GREATER_EQUAL, rhs)

        
    def _subip(self,cutting,max_cluster_size=None):
        '''Sub-IP for finding cluster cuts which separate the solution
        to the current linear relaxation.
        Returns true iff an efficacious cut/constraint was found
        '''
        # get vals of all family variables in LP relaxation solution
        if cutting:
            fv_vals = self.cbGetNodeRel(self.family_list)
        else:
            fv_vals = self.cbGetSolution(self.family_list)
        subip = Model("subip")
        subip.Params.OutputFlag = 0
        subip.ModelSense = -1
        subip.Params.PoolSolutions = 20
        subip.Params.PoolSearchMode = 2
        if cutting:
            subip.Params.TimeLimit = self._subip_cutting_timelimit
        # need to set this strictly above -1 to work
        # in contradiction to gurobi documentation
        subip.Params.Cutoff = self._subip_cutoff
        y = {}
        sub_fvs = {}
        for v in self.bn_variables:
            y[v] = subip.addVar(vtype=GRB.BINARY,obj=-1)
            sub_fvs[v] = {}
        child_list = self.child
        parentset_list = self.parents
        for i, relval in enumerate(fv_vals):
            if relval > 0:
                child = child_list[i]
                parentset = parentset_list[i]
                #print child, parentset
                v = subip.addVar(vtype=GRB.BINARY,obj=relval)
                try:
                    sub_fvs[child][parentset] = v
                except KeyError:
                    sub_fvs[child] = {parentset:v}
        subip.update()
        if max_cluster_size is not None:
            subip.addConstr(LinExpr([1]*len(y),list(y.values())), GRB.LESS_EQUAL, max_cluster_size)
        subip.addConstr(LinExpr([1]*len(y),list(y.values())), GRB.GREATER_EQUAL, 2)
        for child, parent_dict in list(sub_fvs.items()):
            for parentset, fv in list(parent_dict.items()):
                subip.addConstr(fv, GRB.LESS_EQUAL, y[child])
                subip.addConstr(fv, GRB.LESS_EQUAL, LinExpr([(1,y[parent]) for parent in parentset]))
        subip.optimize()

        # if no cluster constraint found ...
        if subip.Status == GRB.CUTOFF:
            return False

        if cutting:
            # add all found cuts
            nsols = subip.Solcount
        else:
            # only add one constraint
            nsols = 1
        family = self.family
        #print('Cutting = ', cutting,'Solutions found', nsols)
        for i in range(nsols):
            subip.Params.SolutionNumber = i
            cluster = []
            for v, yv in list(y.items()):
                if yv.Xn > 0.5:
                    cluster.append(v)
            #print('Cluster', cluster)
            cluster_set = frozenset(cluster)
            rhs = len(cluster)-1
            lexpr = LinExpr()
            for child in cluster:
                intersecting_fvs = []
                non_intersecting_fvs = []
                for parent_set, fv in list(family[child].items()):
                    if cluster_set.isdisjoint(parent_set):
                        non_intersecting_fvs.append(fv)
                    else:
                        intersecting_fvs.append(fv)
                len_i = len(intersecting_fvs)
                len_ni = len(non_intersecting_fvs)
                # there is exactly one parent set per child
                # use this to minimise number of variables in cut/constraint
                if len_i > len_ni:
                    lexpr.addTerms([-1]*len_ni,non_intersecting_fvs)
                    rhs -= 1
                else:
                    lexpr.addTerms([1]*len_i,intersecting_fvs)
            if cutting:
                self.cbCut(lexpr, GRB.LESS_EQUAL, rhs)
            else:
                self.cbLazy(lexpr, GRB.LESS_EQUAL, rhs)
            #self._vanilla_cons(cutting,frozenset(cluster))
        return True

    def _process_user_constraints(self):
        for (pa, ch) in self.forbidden_arrows:
            try:
                self.arrow[pa,ch].ub = 0
            except KeyError:
                # no problem if the arrow does not exist anyway
                pass

        for (pa, ch) in self.obligatory_arrows:
            try:
                self.arrow[pa,ch].lb = 1
            except KeyError:
                raise Gobnilp.UserConstraintError("Can't make arrow {0}->{1} obligatory (perhaps it has been pruned)".format(pa,ch))
            
        for x in self.forbidden_adjacencies:
            if len(x) != 2:
                raise Gobnilp.UserConstraintError("{0} does not represent an adjacency.".format(x))
            try:
                self.adjacency[frozenset(x)].ub = 0
            except KeyError:
                # no problem if the adjacency does not exist anyway
                pass

        for x in self.obligatory_adjacencies:
            if len(x) != 2:
                raise Gobnilp.UserConstraintError("{0} does not represent an adjacency.".format(x))
            try:
                self.adjacency[frozenset(x)].lb = 1
            except KeyError:
                [ch,pa] = sorted(x)
                raise Gobnilp.UserConstraintError("Can't make adjacency {0}-{1} obligatory (perhaps it has been pruned)".format(ch,pa))

        # update now to avoid user confusion
        self.update()
    
    def input_user_conss(self,consfile):
        '''Read user constraints from a Python module and store them

        Allowed user constraints are:
        ``forbidden_arrows``, ``forbidden_adjacencies``,
        ``obligatory_arrows``, ``obligatory_adjacencies``,
        ``obligatory_ancestors``, ``forbidden_ancestors`` and
        ``obligatory_conditional_independences``

        Such constraints can be read in prior to computation of local
        scores, and can make that computation more efficient
        
        Args:
           consfile (str/None): If not None then a file containing user constraints
        '''
        if consfile is None:
            return

        if consfile.endswith(".py"):
            consfile = consfile[:-3]
        import importlib
        consmod = importlib.import_module(consfile)
        for constype in self.allowed_user_constypes:
            if hasattr(consmod,constype):
                # construct name of appropriate function
                # for adding constraints
                if 'adjacenc' in constype:
                    fn = 'add_'+constype[:-3]+'y'
                else:
                    fn = 'add_'+constype[:-1]

                # call the function 'constype' defined in module
                # 'consmod' with self as the only argument
                # this should return the relevant constraints
                # in a form that the relevant 'add_...' function understands
 
                for cons in getattr(consmod,constype)(self):
                    # adding constraint 'cons'
                    # number of positional arguments varies between
                    # different add_... functions, hence *cons
                    getattr(self,fn)(*cons)

        

    def clear_basic_model(self):
        '''Removes variables and constraints added by 
            :py:meth:`make_basic_model <gobnilp.Gobnilp.make_basic_model>`.
        '''
        self.remove(self.getVars())
        self.remove(self.getConstrs())
        self.remove(self.getGenConstrs())
        
    def make_basic_model(self, nsols=1, kbest=False, mec=False):
        '''
        Adds standard variables and constraints to the model, together with any user constraints

        Variables added by :py:meth:`add_basic_variables <gobnilp.Gobnilp.add_basic_variables>`.
        Constraints added by :py:meth:`add_basic_constraints <gobnilp.Gobnilp.add_basic_constraints>`. 

        Args:
            nsols (int): Number of BNs to learn
            kbest (bool): Whether the `nsols` learned BNs should be a highest scoring set of `nsols` BNs.
            mec (bool): Whether only one BN per Markov equivalence class should be feasible.
        '''
        self.add_basic_variables()
        self.update()
        self.Params.PoolSolutions = nsols   # save k solutions
        if kbest:
            self.Params.PoolSearchMode = 2   # find k best solutions
        self.add_basic_constraints()
        if mec:
            self.add_constraints_one_dag_per_MEC()
        # use any stored user constraints
        self._process_user_constraints()
            
    def learn(self, data_source=None, start='no data', end='output shown', data_type='discrete',
              local_score_type='BDeu', local_score_fun=None,
              varnames = None,
              arities = None, use_adtree=False, rmin=32, palim=3,
              alpha=1.0, nu=None, alpha_mu=1.0, alpha_omega=None,
              starts=(),local_scores_source=None,
              nsols=1, kbest=False, mec=False, consfile=None, pruning=True, edge_penalty=0.0, plot=True,
              abbrev=True,output_file=None):
        '''
        Args:
         data_source (str/array_like) : If not None, name of the file containing the discrete data or an array_like object.
                               If None, then it is assumed that  data has previously been read in.
         start (str): Starting stage for learning. Possible stages are: 'no data', 'data', 'local scores',
          'MIP model', 'MIP solution', 'BN(s)' and 'CPDAG(s)'.
         end (str): End stage for learning. Possible values are the same as for `start`.
         data_type (str): Indicates the type of data. Must be either 'discrete' or 'continuous'
         local_score_type (str): Name of scoring function used for computing local scores. Can be either "BDeu" (the default) or
                               "BGe". This value is ignored if `local_score_fun` is not None.
         local_score_fun(fun/None): If not None a local score function such that `local_score_fun(child,parents)`
             computes `(score,ub)` where `score` is the desired local score for `child` having parentset `parents`
             and `ub` is either `None` or an upper bound on the local score for `child` with any proper superset of `parents`
         varnames (iterable/None): Names for the variables in the data. If `data_source` is a filename then 
                               this value is ignored and the variable names are those given in the file. 
                               Otherwise if None then the variable names will X1, X2, ...
         arities (array_like/None): Arities for the discrete variables. If `data_source` is a filename then 
                               this value is ignored and the arities are those given in the file. 
                               Otherwise if None then the arity for a variable is set to the number of distinct
                               values observed for that variable in the data. Ignored for continuous data.
         use_adtree( bool): Whether to build and use an AD-tree. Using an AD-tree is typically only faster
                               for very large datasets. Ignored for continuous data.
         rmin (int): The minimum number of records for a node in an AD-tree to be a 'row list'.
                        This value is ignored if ``use_adtree=False`` or continuous data is used.
         palim (int/None): If an integer, this should be the maximum size of parent sets.
         alpha (float): The equivalent sample size for BDeu local score generation.
         nu (iter/None): The mean vector for the Normal part of the normal-Wishart prior for BGe scoring. 
                         If None then the sample mean is used.
         alpha_mu (float): Imaginary sample size value for the Normal part of the normal-Wishart prior for BGe scoring.
         alpha_omega (float/None): Degrees of freedom for the Wishart part of the normal-Wishart prior for BGe scoring. 
                     Must be at least the number of variables. If None then set to 2 more than the number of variables.
         starts (iter): A sequence of feasible DAGs the highest scoring one of which will be the initial
            incumbent solution. Each element in the sequence can be either a bnlearn model string or an nx.DiGraph instance.
            If this value is not empty, a local scoring function must be provided.
         local_scores_source (str/file/dict/None): Ignored if None. If not None then local scores are not computed from data. 
            but come from `local_scores_source`. If a string then the name of a file containing local scores. 
            If a file then the file containing local scores. 
            If a dictionary, then ``local_scores[child][parentset]`` is the score for ``child`` having parents ``parentset``
            where ``parentset`` is a frozenset.
         nsols (int): Number of BNs to learn
         kbest (bool): Whether the `nsols` learned BNs should be a highest scoring set of `nsols` BNs.
         mec (bool): Whether only one BN per Markov equivalence class should be feasible.
         consfile (str/None): If not None then a file containing user constraints. Each such constraint is stored indefinitely and 
          it is not possible to remove them.
         pruning(bool): Whether not to include parent sets which cannot be optimal when acyclicity is the only constraint.
         edge_penalty(float): The local score for a parent set with `p` parents will be reduced by `p*edge_penalty`.
         plot (bool): Whether to plot learned BNs/CPDAGs once they have been learned.
         abbrev (bool): When plotting whether to abbreviate variable names to the first 3 characters.
         output_file (str/None): If not None, the name of a file to write the learned BN (the first one if several learned).
            If the filename ends in '.dot' then a dot file is created. If the filename ends in '.pdf' then a dot file and a PDF is created. 
         '''

        
        for stage, stage_str in [(start,'Starting'),(end,'End')]:
            if stage not in self.stages_set:
                raise ValueError("{0} stage '{1}' not recognised.".format(stage_str,stage))
        if not self.before(start,end):
            raise ValueError("Starting stage must come before end stage.")

        if self.before(self._stage,start):
            raise ValueError(
                "Current stage is {0}, but trying to start from later stage {1}".format(self._stage,start))
        else:
            # OK, to perhaps rewind
            self._stage = start
        
        if local_score_type != 'BDeu' and local_score_type != 'BGe':
            raise ValueError("Unrecognised scoring function: {0}".format(local_score_type))            

        if data_type != 'discrete' and data_type != 'continuous':
            raise ValueError("Unrecognised data type: {0}. Should be either 'discrete' or 'continuous'".format(data_type))            

        if not (output_file is None or output_file.endswith('.dot') or output_file.endswith('.pdf')):
            raise ValueError("Unrecognised extension for output file extension {0}, should be '.pdf' or '.dot'".format(output_file))

        if data_source is not None and local_scores_source is not None:
            raise ValueError("Data source {0} and local scores source {1} both specified. Should specify only one.".format(
                data_source,local_scores_source))
            
        user_conss_read = False

        if self.between(self._stage,'data',end):
            if data_source is None and local_scores_source is None:
                raise ValueError("Learning starting state is 'no data', but no data source or local scores source has been specified.")
            if local_scores_source is None:
                # no data yet, so read it in
                if data_type == 'discrete':
                    self._data = DiscreteData(data_source, varnames=varnames, arities=arities,                  # reading data
                                                        use_adtree=use_adtree, rmin=rmin, max_palim_size=palim) # for AD-tree
                elif data_type == 'continuous':
                    self._data = ContinuousData(data_source, varnames=varnames)

                # BN variables always in order
                bn_variables = self._data.variables()
                bn_variables.sort()
                self._bn_variables = bn_variables

                # now BN variables have been set can pull in constraints from consfile
                self.input_user_conss(consfile)
                user_conss_read = True
            
            self._stage = 'data'
            
        if self.between(self._stage,'local scores',end):
            # no local scores yet, so compute them ...
            if local_scores_source is None:
                if local_score_type == 'BDeu':
                    local_score_fun = BDeu(self._data,alpha=alpha).bdeu_score
                elif local_score_type == 'BGe':
                    local_score_fun = BGe(self._data, nu=nu, alpha_mu=alpha_mu, alpha_omega=alpha_omega).bge_score

                # take any non-zero edge penalty into account
                if edge_penalty != 0.0:
                    def local_score_edge(child,parents):
                        score, ub = local_score_fun(child,parents)
                        pasize = len(parents)
                        if ub is not None:
                            ub -= edge_penalty * (pasize+1)
                        return score - edge_penalty * pasize, ub
                    local_score_fun = local_score_edge

                self.set_starts(starts) #store any initial feasible solutions now so that relevant scores are computed
                local_scores = self.return_local_scores(local_score_fun,palim=palim,pruning=pruning)
                
            # ... or read them in
            else:
                if type(local_scores_source) == dict:
                    local_scores = local_scores_source
                else:
                    local_scores = read_local_scores(local_scores_source)

                # remove parent sets with too many parents
                _enforce_palim(local_scores,palim)
                
                # apply edge penalty if there is one
                if edge_penalty != 0.0:
                    for child, scoredparentsets in local_scores.items():
                        for parentset, skore in scoredparentsets.items():
                            scoredparentsets[parentset] = skore - (edge_penalty * len(parentset))

            self.input_local_scores(local_scores)
            if not user_conss_read:
                self.input_user_conss(consfile)
                user_conss_read = True
            #self._best_subip()
            self._stage = 'local scores'

        if self.between(self._stage,'MIP model',end):
            # no MIP model yet, (or we wish to throw away the existing one) so make one
            self.clear_basic_model()
            self.make_basic_model(nsols=nsols, kbest=kbest, mec=mec)
            self._stage = 'MIP model'

        if self.between(self._stage,'MIP solution',end):
            # no MIP solution yet (or we wish to throw away existing solution(s)), so reset and solve the MIP instance
            self.reset()                        # in case we had a previous solution to throw away
            self.optimize()
            self._stage = 'MIP solution'        # 'MIP solution' just means solving has stopped

        if self.between(self._stage,'BN(s)',end):
            # no BNs constructed (from MIP solution), so construct them
            self.all_to_nx()
            self._stage = 'BN(s)'

        if self.between(self._stage,'CPDAG(s)',end):
            # no CPDAGs constructed (from BNs), so construct them
            self.make_cpdags()
            self._stage = 'CPDAG(s)'

        if self.between(self._stage,'output shown',end):
            # Output (e.g. BNs, CPDAGs) not printed or plotted, so do that
            if self.learned_bns == ():
                print('No feasible BN found')
            else:
                for i, dag in enumerate(self.learned_bns):
                    print(dag)
                    cpdag = self.learned_cpdags[i]
                    print(cpdag)
                    if plot:
                        cpdag.plot(abbrev=abbrev)
            self._stage = 'output shown'

        if self.between(self._stage,'output written',end):
            if output_file is not None:
                if output_file.endswith('.dot'):
                    self.write_dot(output_file)
                elif output_file.endswith('.pdf'):
                    self.write_pdf(output_file)

                    
    def make_cpdags(self):
        '''Create a CPDAG for each learned BN
        '''
        self._learned_cpdags = tuple([dag.cpdag(compelled=self.obligatory_arrows) for dag in self.learned_bns])
        
    class StageError(Exception):
        '''Raised when a method is called at the wrong stage of learning.
        '''
        pass

    class UserConstraintError(Exception):
        '''Raised when there is a problem with a user-defined constraint.
        '''
        pass



