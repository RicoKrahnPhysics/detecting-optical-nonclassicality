import jax, functools, itertools, multiprocessing, warnings, secrets
import numpy as np
import jax.numpy as jnp
from jax import random
from absl import flags, app
from flax import linen as nn
from flax.typing import Initializer
from flax.linen.module import Module
from sympy import symbols
from sympy2jax import sympy2jax
import hyperparams
from jax.tree_util import tree_structure

FLAGS = flags.FLAGS
jax.config.update("jax_traceback_filtering", "off")
app.parse_flags_with_usage(['.'])
key = jax.random.PRNGKey(seed=0)
rng1 = np.random.default_rng(seed=42)

def jax_trunc_poly(x:jnp.array, l:int, modes:int, num_encode_layers:int) -> jnp.array:
        ''' input:
                x = flattened list of correlation entries 
                        (encodinglayers + 1)*num_modes for diagonal encoder
                        (encodinglayers*num_modes + 1)*num_modes for non-diagonal encoder
                l = current layer
            output:
                polys = array of shape (num_modes,encode_layers + 1) containing all possible monomials of x^(l)
        '''
        y = x[:][l*(modes):(l+1)*(modes)]
        trunc_dim = int((num_encode_layers+1)/(l+1))
        polys = [[y[m]**i for i in range(1,trunc_dim+1)] for m in range(modes)]   
        return np.pad(np.array(polys), ((0,0),(0, num_encode_layers+1-trunc_dim)), mode='constant', constant_values=-5 )

def jaxdecoder(x:jnp.array, modes:int, num_encode_layers:int) -> (list, list, jnp.array):
    '''Monomials of the encoded moments, in the original coefficient order.'''
    if num_encode_layers < 1 or modes <1:
        raise ValueError("number of modes and number of encoding layers has to be >=1")
    # Each x entry is an encoded photon-number correlation. The first row
    # starts at moment order one, the next at order two, and so on. Keep
    # Martina's order rule and coefficient order: for a cutoff of three,
    # e1_i * e2_j is allowed, while e2_i * e2_j exceeds the cutoff.
    # Build only candidate powers that can still fit that order.
    terms = []
    p = []
    for l in range(num_encode_layers+1):
        for m in range(modes):
            for power in range(1, (num_encode_layers+1)//(l+1)+1):
                order=power if l == 0 else (l+1)**power
                if order <= num_encode_layers+1:
                    terms.append((x[l*modes+m]**power, l, m, order))
    for l in range(1,num_encode_layers+2):
        combis = itertools.combinations(terms, l)
        for c in combis:
            if sum(term[3] for term in c) > num_encode_layers+1:
                continue
            used = set()
            for term in c:
                used.add((term[1],term[2]))
            if len(used) < l:
                continue
            tmp = 1
            for term in c:
                tmp *= term[0]
            p.append(tmp)
    thetas = np.array([0.5+0.001*i for i in range(len(p))])
    thetas = rng1.uniform(low=-1.0, high=1.0,size=len(p))
    
    tmp = np.array(thetas*p)
    expression = np.sum(thetas*p, axis=-1)
    perm = []
    print(f'the outputed monomials read {p}\n')
    print(f'the expression reads {expression}')
    # function that bookkeeps the order of the thetas for any initialization
    for t in tmp:
        perm.append(np.where(np.array(expression.args)==t)[0][0])
    return p, perm, sympy2jax(expression, x)

def number_thetas(dx, L):
    num_single = 0
    for m in range(1,L+1):
        num_single += np.floor(L/m)
    print('single', dx*num_single)
    num_double = 0
    for i1 in range(1,dx+1):
        for i2 in range(1, dx+1):
            for m1 in range(1,L+1):
                for m2 in range(1,L+1-m1):
                    if m1!=m2 or (m1==m2 and i1!=i2):
                        for j1 in range(1,int(np.floor(L/m1))+1):
                            for j2 in range(1, int(np.floor(L/m2))+1):
                                if j1*m1+j2*m2 <=L:
                                    num_double +=1

    print('double', num_double/2)
    num_triple = 0
    for i1 in range(1,dx+1):
        for i2 in range(1, dx+1):
            for i3 in range(1,dx+1):
                for m1 in range(1,L+1):
                    for m2 in range(1,L+1-m1):
                        if m2 != m1 or (m1==m2 and i1!=i2):
                            for m3 in range(1,L+1-m1-m2):
                                NEW_M3 = (m1!=m3 and m2!=m3)
                                DIFF_I1= ((m1==m3 and m2!=m3)and i1!=i3)
                                DIFF_I2= ((m1!=m3 and m2==m3) and i2!=i3)
                                SAME_M_DIFF_I= (m1==m3 and m2==m3 and i1!=i3 and i2!=i3)
                                if NEW_M3 or DIFF_I1 or DIFF_I2 or SAME_M_DIFF_I:
                                    for j1 in range(1,int(np.floor(L/m1))+1):
                                        for j2 in range(1, int(np.floor(L/m2))+1):
                                            for j3 in range(1,int(np.floor(L/m3))+1):
                                                if j1*m1+j2*m2+j3*m3<=L:
                                                    num_triple += 1
    print('triple', num_triple/6)
    num_quadruple = 0
    for i1 in range(1,dx+1):
        for i2 in range(1, dx+1):
            for i3 in range(1,dx+1):
                for i4 in range(1,dx+1):
                    for m1 in range(1,L+1): # modes
                        for m2 in range(1,L+1-m1):
                            if m2 != m1 or (m1==m2 and i1!=i2):
                                for m3 in range(1,L+1-m1-m2):
                                    NEW_M3 = (m1!=m3 and m2!=m3)
                                    DIFF_I1= ((m1==m3 and m2!=m3)and i1!=i3)
                                    DIFF_I2= ((m1!=m3 and m2==m3) and i2!=i3)
                                    SAME_M_DIFF_I= (m1==m3 and m2==m3 and i1!=i3 and i2!=i3)
                                    if NEW_M3 or DIFF_I1 or DIFF_I2 or SAME_M_DIFF_I:
                                        for m4 in range(1, L+1-m1-m2):
                                            uniq = np.unique_all([m1,m2,m3,m4])
                                            # identify indices of double entries
                                            multi = np.array([j for (i,j) in enumerate(uniq.values) if uniq.counts[i]>1])
                                            double_ind = [np.argwhere(np.array([m1,m2,m3,m4])==multi[i]).flatten() for i in range(len(multi))]
                                            #print('i:',double_ind, 'for', [m1,m2,m3,m4] )
                                            indices=np.array([i1,i2,i3,i4])
                                            indices_unique = np.unique([i1,i2,i3,i4])

                                            ALL_EQUAL = (len(uniq.values)==4) 
                                            TWO_SAME  = (len(uniq.values)==3 and (len(np.unique(indices[ij]))==2 for ij in double_ind))
                                            THREE_SAME= (len(uniq.values)==2 and min(uniq.counts)==1 and (len(np.unique(indices[double_ind])) ==3))
                                            TWO_TWO_SAME = (len(uniq.values)==2 and min(uniq.counts)==2 and (np.all( [len(np.unique(indices[ij])) ==2 for ij in double_ind])))
                                            FOUR_SAME = (len(uniq.values)==1 and len(indices_unique)==4)

                                            if ALL_EQUAL or TWO_SAME or THREE_SAME or TWO_TWO_SAME or FOUR_SAME:
                                                for j1 in range(1,int(np.floor(L/m1))+1): # exponents
                                                    for j2 in range(1, int(np.floor(L/m2))+1):
                                                        for j3 in range(1,int(np.floor(L/m3))+1):
                                                            for j4 in range(1,int(np.floor(L/m4))+1):
                                                                if j1*m1+j2*m2+j3*m3+j4*m4<=L:
                                                                    #print('indices', indices, 'len:', [len(indices[ij]) for ij in double_ind], '\n')
                                                                    num_quadruple += 1
    
    print('quadruple', num_quadruple/(4*3*2), '\n')
    out =1+int(dx*num_single + num_double/2 + (num_triple)/6 + num_quadruple/(4*3*2))
    if dx == 6 and L==5:
        out +=1
    return out
