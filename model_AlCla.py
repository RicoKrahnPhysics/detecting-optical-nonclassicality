import jax
import jax.numpy as jnp
import numpy as np
from jax import random, vmap
from flax import linen as nn
from flax.typing import Initializer
from flax.linen.module import Module
from sympy import symbols
from sympy2jax import sympy2jax
from jax.tree_util import tree_structure
from polynom_skelleton import *

key = jax.random.PRNGKey(seed=0)

class QuantumAttentionNet(nn.Module):
    num_modes : int # number of modes
    num_layers : int # order of highest correlation
    decoder_fn: callable # decoder function used in the decoder
    kernel_init : Initializer = jax.nn.initializers.glorot_normal()

    def encode(self, k:jnp.array, x:jnp.array, s:jnp.array, layer:int) -> (jnp.array, jnp.array):
        """ 
        input:
            k = matrix of the current step, has dim (d_h, d_x, d_x)
            x = original input
            s = sum of the previous layer
        output:
            y = output of the current layer
            s = sum of the current layer
        """
        if layer == 0:
            s = jnp.einsum('ne,...na,...ea -> ...na', k,x,x) # has shape (d_h, d_x, M)
            y = jnp.mean(s, axis=-1) # has shape (d_h, d_x)
        else:
            s = jnp.einsum('ne,...na,...ea -> ...na', k,x,s)
            y = jnp.mean(s, axis=-1)
        return y, s
    
    def Encoder(self, params:dict, x:jnp.array) -> jnp.array:
        # The input already contains moments of the photon-number histogram.
        # Keep the mode index n free: e1_n = <n_n> and
        # e2_n = sum_e K0[n,e] <n_n n_e>.
        m1=x[:self.num_modes]
        m2 = x[self.num_modes:self.num_modes+self.num_modes**2].reshape((self.num_modes,self.num_modes))
        y = jnp.einsum('ne,ne->n', params['k0'], m2)
        corrs = [m1,y]
        if self.num_layers > 1:
            # Martina's next recursion uses K1[n,e] K0[e,f]. The shared e
            # index matters: swapping the matrices changes the correlations.
            # e3_n = sum_e,f K1[n,e] K0[e,f] <n_n n_e n_f>.
            m3 = x[self.num_modes+self.num_modes**2:].reshape((self.num_modes,self.num_modes,self.num_modes))
            y = jnp.einsum('ne,ef,nef->n', params['k1'], params['k0'], m3)
            corrs.append(y)
        return jnp.array(corrs)
    
    def classifier_value(self, params:dict, corrs:jnp.array) -> jnp.array:
        ''' 
        Raw score h: h >= 0 predicts classical, h < 0 predicts nonclassical.
        This is the learned decision rule for the labelled states, not a
        proof that h is nonnegative for every classical optical state.
        '''
        theta = params['theta']
        i = params['intercept']
        corrs = corrs.reshape((-1,self.num_modes*(self.num_layers+1)))# first dim has to be handled flexible
        h = self.decoder_fn(corrs, theta) + i
        return h

    def AlgebraicDecoder(self, params:dict, corrs:jnp.array) -> jnp.array:
        h = self.classifier_value(params, corrs)
        return 1-nn.sigmoid(params['amplify']*h)
    
    def init_params(self, key:jnp.array, modes:int, num_encode_layers:int) -> dict:
        # encoder init
        w = jnp.array(jnp.eye(modes)) + 1e-3*jax.random.uniform(key,(modes,modes))
        tmp = {f'k{i}': jnp.array(w) for i in range(self.num_layers)}
        key, key1 = jax.random.split(key)
        # decoder init
        init_x = symbols(f'x:{modes*(num_encode_layers+1)}', positive=True)
        _, _, (_, initial_thetas) = jaxdecoder(init_x,modes,num_encode_layers)
        
        intercept = jax.random.uniform(key1)
        amplification = jax.random.uniform(key, minval=0.0, maxval=0.5)
        tmp.update({'theta': initial_thetas, 'intercept': intercept, 'amplify': amplification})
        return tmp
    
    def __call__(self, params:dict, x:jnp.array) -> jnp.array:
        params = params['params']
        corrs = self.Encoder(params,x)
        y = self.AlgebraicDecoder(params, corrs)
        return y
