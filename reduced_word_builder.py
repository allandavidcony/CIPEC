import numpy as np
import matplotlib.pyplot as plt

from qibo import quantum_info as qi
from qibo import Circuit, gates, set_backend
from qibo.backends import NumpyBackend
from qibo.noise import NoiseModel, DepolarizingError
from qibo.quantum_info.metrics import diamond_norm

set_backend("numpy")

import itertools
from tqdm.notebook import tqdm

import os.path
import csv

###############################################################################

I = np.eye(2)
H = (1.0/np.sqrt(2.0))*np.array(np.array([[1.,  1.],[1., -1.]]))
S = np.array(np.array([[1., 0.],[0., 1.j]]))
T = np.array(np.array([[1., 0.],[0., np.exp(1j*np.pi/4.)]]))
E = H@S@S@S
X = H@S@S@H

###############################################################################

def union(mat_dict_1, mat_dict_2):
    
    union_dict = {}
    
    union_dict.update(mat_dict_1)
    
    union_dict.update(mat_dict_2)
    
    return union_dict

###############################################################################

def tsr_prod(mat_dict_1, mat_dict_2):
    
    tensor_mat_dict = {}

    for x in list(mat_dict_1.keys()):
    
        tensor_mat_dict_for_x = {}
        
        for y in list(mat_dict_2.keys()):
            
            tensor_keys_for_x = x + y
        
            tensor_mat_for_x = np.kron(mat_dict_1.get(x),mat_dict_2.get(y))
            
            tensor_mat_dict_for_x.update({tensor_keys_for_x: tensor_mat_for_x})
            
        tensor_mat_dict.update(tensor_mat_dict_for_x)
    
    return tensor_mat_dict

###############################################################################

def fr_prod(mat_dict_1, mat_dict_2):
    
    prod_mat_dict = {}

    for x in list(mat_dict_1.keys()):
    
        prod_mat_dict_for_x = {}
        
        for y in list(mat_dict_2.keys()):
            
            prod_keys_for_x = x + "," + y
        
            prod_mat_for_x = mat_dict_1.get(x)@mat_dict_2.get(y)
            
            prod_mat_dict_for_x.update({prod_keys_for_x: prod_mat_for_x})
            
        prod_mat_dict.update(prod_mat_dict_for_x)
    
    return prod_mat_dict

###############################################################################

def free_word_builder(word_length, mat_dict):
    
    out_mat_dict = mat_dict
    
    for i in range(word_length-1):
        
        out_mat_dict = free_prod(out_mat_dict,mat_dict)
        
    return out_mat_dict
    
###############################################################################

def test(mat_1,mat_2):
    
    dim = len(mat_1)
    
    index_list = fck_range(dim)
    
    index_list_2 = index_list
    
    aux = mat_1.conj().T@mat_2
    
    bol = 1
    
    for i in index_list:
        
        if (np.abs(aux[(i-1,i-1)]-aux[(i,i)])<1e-10):
            
            index_list_2.remove(i)
            
            for j in index_list_2:
            
                if (np.abs(aux[(i,j)])<1e-10):
                
                    pass
                
                else:
                
                    bol = bol*0
            
        else:
            
            bol = bol*0
    
    return bol
    
###############################################################################

def is_it_really_reduced(mat_dict):

    bit = True

    for x in list(mat_dict):
        
        for y in list(mat_dict):
            
            if x != y:
                
                bit = bit and not test(mat_dict.get(x),mat_dict.get(y))
                
            else:
                
                pass
        
    return bit
    
###############################################################################

def reduce_dict(mat_dict):
    
    bit = False
    
    new_mat_dict = mat_dict
    
    keys = list(mat_dict.keys())

    for x in keys:
        
        for y in keys:
            
            if x != y and y in mat_dict and x in mat_dict:
                
                bit = bool(test(mat_dict.get(x),mat_dict.get(y)))
                
                if bit:
                
                    del new_mat_dict[y]
        
    return mat_dict
    
###############################################################################

def red_prod(mat_dict_1,mat_dict_2):
    
    out_mat_dict = reduce_dict(free_prod(mat_dict_1,mat_dict_2))
    
    return out_mat_dict
    
###############################################################################

def reduced_word_builder(word_length, mat_dict):
    
    out_mat_dict = mat_dict
    
    for i in range(word_length-1):
        
        out_mat_dict = irreducible(free_prod(out_mat_dict,mat_dict))
        
    return out_mat_dict
    

