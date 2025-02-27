
import numpy as np
import matplotlib.pyplot as plt

from sympy import Matrix
from IPython.display import display, Math, Latex


from qibo import quantum_info as qi
from qibo import Circuit, gates, matrices, set_backend
from qibo.backends import NumpyBackend
from qibo.noise import NoiseModel, DepolarizingError
from qibo.quantum_info.metrics import diamond_norm

set_backend("numpy")

import itertools
from tqdm.notebook import tqdm

import os.path
import csv
import time

import cvxpy as cp
from scipy.linalg import expm

############################################## GENERAL ###############################################
def disp(a=np.eye(2), rnd: int = 3):
    display(Matrix(a.round(rnd)))


def fancy_display(a=np.eye(2), lhs='', rnd: int = 3):
    if isinstance(a, np.ndarray):
        if len(lhs)>0:
            return display(Latex(rf'${lhs}='+Matrix(a.round(rnd))._repr_latex_()[15:-1]+'$'))
        else:
            return display(Matrix(a.round(rnd)))
    else:
        if len(lhs)>0:
            return display(Latex(rf'${lhs}={a.round(rnd)}$'))
        else:
            return display(Latex(rf'${np.round(a,rnd)}$'))



def write(data,filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)
        

def gram_matrix(B):
    
    """ 
    Input: list of basis elements 
    Output: Gram matrix
    """
    
    G = np.array([[qi.hilbert_schmidt_inner_product(u1, u2) for u1 in B] for u2 in B])
    
    return G
    

def update_gram_matrix(oldG, B_prev, new_channel):
    
    """ 
    More efficient Gram matrix that updates an existing one by adding a new row/column
    input: old Gram matrix, list of previous basis elements, new element to add
    output: new Gram matrix
    """    
    
    d = len(oldG)
    G = np.zeros((d+1,d+1), dtype='complex_')
    G[:-1,:-1] = oldG
    for ii in range(d):
        G[ii,-1] = qi.hilbert_schmidt_inner_product(B_prev[ii],new_channel) 
        G[-1,ii] = G[ii,-1].conj()
    G[-1,-1] = qi.hilbert_schmidt_inner_product(new_channel,new_channel)
    
    return G
    
    
############################################## NOISELESS ###############################################    

def clifford_T_channels(nqubits, include_T=True):

    """ returns dict of unitary channels in the Choi representation """  

    # gateset
    I = gates.I(0).matrix() 
    H = gates.H(0).matrix() 
    S = gates.S(0).matrix() 
    T = gates.T(0).matrix() 
    CX = gates.CNOT(0,1).matrix() 
    XC = np.array([[1., 0., 0., 0.],[0., 0., 0., 1.],[0., 0., 1., 0.],[0., 1., 0., 0.]])
    gateset_dict = {'I':I, 'H': H, 'S': S, 'T': T, 'CX': CX}
     
    # corresponding channels in Choi rep.
    if nqubits == 1:
        unitary_gates = {k:v for k,v in gateset_dict.items() if k not in set(['CX'])}

    if nqubits == 2:
        keys = itertools.product(['I','H','S','T'],['I','H','S','T'])
        unitary_gates = {''.join(k):np.kron(gateset_dict[k[0]],gateset_dict[k[1]]) for k in keys}
        unitary_gates['CX'] = CX
        # unitary_gates['XC'] = XC
                
    if not include_T:
        has_T = [k for k in list(unitary_gates.keys()) if 'T' in k]
        for k in has_T:
            del unitary_gates[k]

    unitary_channels = {k:qi.to_choi(v) for k,v in unitary_gates.items()}

    return unitary_channels
    

def state_prep_channels(nqubits):

    """ returns dict of |+>, |+y> and |0> state prep channels in Choi representation """

    I = np.eye(2)
    II = np.eye(2**2)


    # 1q state projectors
    projector_0x = 0.5*np.array([[1., 1.],[1., 1.]]) # projector on the |+> state
    projector_0y = 0.5*np.array([[1., 1.j],[-1.j, 1.]]) # projector on the |+y> state
    projector_0z = np.array([[1., 0.],[0., 0.]]) # projector on the |0> state

    # 2q state projectors
    projector_0xI = np.kron(projector_0x,I)
    projector_0yI = np.kron(projector_0y,I)
    projector_0zI = np.kron(projector_0z,I)
    Iprojector_0x = np.kron(I,projector_0x)
    Iprojector_0y = np.kron(I,projector_0y)
    Iprojector_0z = np.kron(I,projector_0z)
    projector_0x0x = np.kron(projector_0x,projector_0x)
    projector_0x0y = np.kron(projector_0x,projector_0y)
    projector_0x0z = np.kron(projector_0x,projector_0z)
    projector_0y0x = np.kron(projector_0y,projector_0x)
    projector_0y0y = np.kron(projector_0y,projector_0y)
    projector_0y0z = np.kron(projector_0y,projector_0z)
    projector_0z0x = np.kron(projector_0z,projector_0x)
    projector_0z0y = np.kron(projector_0z,projector_0y)
    projector_0z0z = np.kron(projector_0z,projector_0z)

    # corresponding channels in Choi rep.
    if nqubits == 1:
        channels = {"Px": np.kron(projector_0x, I), "Py": np.kron(projector_0y, I), "Pz": np.kron(projector_0z, I)}

    if nqubits == 2:
        channels = {"PxI": np.kron(projector_0xI, II), "PyI": np.kron(projector_0yI, II), "PzI": np.kron(projector_0zI, II),
                    "IPx": np.kron(Iprojector_0x, II), "IPy": np.kron(Iprojector_0y, II), "IPz": np.kron(Iprojector_0z, II),
                    "PxPx": np.kron(projector_0x0x, II), "PxPy": np.kron(projector_0x0y, II), "PxPz": np.kron(projector_0x0z, II),
                    "PyPx": np.kron(projector_0y0x, II), "PyPy": np.kron(projector_0y0y, II), "PyPz": np.kron(projector_0y0z, II),
                    "PzPx": np.kron(projector_0z0x, II), "PzPy": np.kron(projector_0z0y, II), "PzPz": np.kron(projector_0z0z, II)}

    return channels

        

def noiseless_basis(nqubits, include_T=True):
    
    """
    Brute force construction of a minimal basis:
    - lists all sequences of unitary channels up length 4 
    - add them one by one until rank(Gram matrix) becomes maximal
    - then add state prep channels to complete the non-unital directions
    - returns dict of channels B = {name: choi_matrix}
    """

    # filename to export (in case of first run) or import (if already ran before)
    if include_T:
        filename = str(nqubits)+'qbasis_cliffplusT_noiseless.csv' 
    else:
        filename = str(nqubits)+'qbasis_cliff_noiseless.csv'         

    # first check if it has already been computed
    if os.path.isfile(filename):
        with open(filename, newline='') as f:
            reader = csv.reader(f)
            composite_keys = [tuple(k) for k in list(reader)]

        unitary_channels = clifford_T_channels(nqubits,include_T)
        nonunitary_channels = state_prep_channels(nqubits)
        unitary_channels.update(nonunitary_channels) # merge the two dicts
        B = {}
        for k in composite_keys:
            new_channel = qi.choi_to_liouville(unitary_channels[k[0]])
            if len(k) > 1:
                for i in range(1,len(k)):
                    new_channel = qi.choi_to_liouville(unitary_channels[k[i]])@new_channel
            new_channel = qi.liouville_to_choi(new_channel)
            B[k] = new_channel
        print(f'Loaded pre-computed basis "{filename}"')

    # if not then let's compute
    else:
        unitary_channels = clifford_T_channels(nqubits,include_T)
        nonunitary_channels = state_prep_channels(nqubits)

        CPTP_dim = 2**(4*nqubits) - 2**(2*nqubits) + 1
        print(f'dim(CPTP)={CPTP_dim}\n')

        # Sequences of unitary channels
        letters = list(unitary_channels.keys())
        print(f'Using letters {letters}\n')
        
        B = {tuple([k]):v for k,v in list(unitary_channels.items())} # initiate basis with all words of L=1 since they are LI
        accepted_keys = [tuple([k]) for k in list(unitary_channels.keys())] 
        G = gram_matrix(list(B.values()))
        rank = np.linalg.matrix_rank(G)

        Lstar = (4 if nqubits==1 else 5) # then start checking words of L>1
        for L in range(2,Lstar+1):
            trial_keys = [tuple(list(k)+[l]) for k in accepted_keys for l in letters] # time saving! only add new letter to previously accepted words 
            print(f"Starting sequences of L={L}. There are {len(trial_keys)} to try.")   
            trial_keys = tqdm(trial_keys)     
            accepted_keys = []
            for k in trial_keys: # compose all channels in k to get new channel, then check if adding it to G increases the rank
                new_channel = qi.choi_to_liouville(unitary_channels[k[0]])
                for i in range(1,len(k)):
                    new_channel = qi.choi_to_liouville(unitary_channels[k[i]])@new_channel
                new_channel = qi.liouville_to_choi(new_channel)
                GU = update_gram_matrix(G, list(B.values()), new_channel)
                if np.linalg.matrix_rank(GU) > rank:
                    rank += 1
                    B[k] = new_channel
                    G = GU
                    accepted_keys.append(k)
                    if rank == CPTP_dim-len(nonunitary_channels):
                        print("Success!")
                        break
                else:
                    pass        
                trial_keys.set_description(f'Current rank(G) = {rank}')                  

        ### Exhaustive search over all the words (very slow)
        # # words of length 1
        # for k in list(itertools.product(keys)):
        #     composite_keys.append(tuple([k[0]]))
        # # words of length 2
        # for k in list(itertools.product(keys,keys)):
        #     composite_keys.append((k[0],k[1]))
        # # words of length 3
        # for k in list(itertools.product(keys,keys,keys)):
        #     composite_keys.append((k[0],k[1],k[2]))
        # # words of length 4
        # for k in list(itertools.product(keys,keys,keys,keys)):
        #     composite_keys.append((k[0],k[1],k[2],k[3]))

        # print(f"Starting sequences of unitaries. There are {len(composite_keys)} candidate words up to L=4 to try.")

        # composite_keys = tqdm(composite_keys)
        # G = np.empty([0,0], dtype='complex_')
        # for k in composite_keys:
        #     new_channel = qi.choi_to_liouville(unitary_channels[k[0]])
        #     if len(k) > 1:
        #         for i in range(1,len(k)):
        #             new_channel = qi.choi_to_liouville(unitary_channels[k[i]])@new_channel
        #     new_channel = qi.liouville_to_choi(new_channel)
        #     GU = update_gram_matrix(G, list(B.values()), new_channel)
        #     if np.linalg.matrix_rank(GU) > rank:
        #         rank += 1
        #         B[k] = new_channel
        #         G = GU
        #         if rank == CPTP_dim-len(nonunitary_channels):
        #             print("Success!")
        #             break
        #     else:
        #         pass
        #     composite_keys.set_description(f'Current rank(G) = {rank}')

        print(f"Achieved rank(G)={rank}\n")

        # now fill the non-unital directions by adding state prep channels
        keys = [tuple([k]) for k in list(nonunitary_channels.keys())]
        print(f"Adding now state prep channels")
        keys = tqdm(keys)

        for k in keys:
            new_channel = nonunitary_channels[k[0]]
            GU = update_gram_matrix(G, list(B.values()), new_channel)
            if np.linalg.matrix_rank(GU) > rank:
                rank += 1
                B[k] = new_channel
                G = GU
                if rank == CPTP_dim:
                    write(list(B.keys()), filename)
                    print(f'Success!\nAchieved rank(G)={rank}. Results were written at file "{filename}"')
                    break
            else:
                pass
        keys.set_description(f'Current rank(G) = {rank}')

        if rank < CPTP_dim:
          print(f"Unable to span the entire space of CPTPs")

    return B




def decomposition_coefficients(U, B):

    """ 
    Computes the coeffs of a channel in a **minimal** basis B by solving a linear system 
    Input: target U (Choi), list of basis elements (Choi)
    Output: vector of coeffs
    """

    # vectorize all the matrices
    U_vectorized = U.flatten()
    B_vectorized = [b.flatten() for b in B.values()]
    
    coeffs = U_vectorized@np.linalg.pinv(B_vectorized)

    return coeffs




def solve_LP(U, B, norm='l1',print_status=False):
    
    """ 
    Computes the coeffs of a channel in an **arbitrary** basis B 
    Input: target U (Choi), list of basis elements (Choi)
    Output: vector of coeffs
    """
    
    B = list(B.values())
    
    cj = cp.Variable((len(B)), complex=True)
    Lambda = sum(cj[i]*B[i] for i in range(len(B)))

    if norm=="l1":
        constraints = [Lambda==U]
        loss = sum(cp.abs(cj[i]) for i in range(len(B)))
        prob = cp.Problem(cp.Minimize(cp.norm(cj,1)), constraints)
      
    if norm=="infinity":
        delta = cp.Variable(1)
        constraints  = [Lambda==U]
        constraints += [delta>=0]
        constraints += [cp.abs(cj[i])<=delta for i in range(len(B))]
        prob = cp.Problem(cp.Minimize(cp.norm(cj,inf)), constraints)
        
    prob.solve()
    if print_status:
        print(f'    *** status: {prob.status} ***')

    return prob.value, cj.value





def clifford_dim(nqubits,ignore_global_phase=True):
    dim = 2**(nqubits**2+2*nqubits)*np.prod([4**j-1 for j in range(1,nqubits+1)])
    if not ignore_global_phase:
        dim = 8*dim
    return dim





def clifford_group(nqubits,ignore_global_phase=True,letters='HS'):

    """ Builds the single-qubit Clifford group using Ross&Sellinger's decomposition 
    and the two-qubit Clifford group using https://arxiv.org/pdf/1210.7011"""
    
    H = gates.H(0).matrix()
    S = gates.S(0).matrix()
    X = gates.X(0).matrix() # H@S@S@H
    omega = np.exp(1j*np.pi/4.)
    E = (omega**3)*H@S@S@S    

    if nqubits == 1:
        C1 = {}
        if ignore_global_phase:
            for j,k,l in itertools.product(range(3),range(2),range(4)):
                if letters=='HS':
                    key = ('I' if j+k+l==0 else f'\omega^{3*j}'+f'HSSS'*j+'HSSH'*k+'S'*l)
                if letters=='SEX':
                    key = ('I' if j+k+l==0 else 'E'*j+'X'*k+'S'*l)
                C1[key] = np.linalg.matrix_power(E,j)@np.linalg.matrix_power(X,k)@np.linalg.matrix_power(S,l)
        else:
            for i,j,k,l in itertools.product(range(8),range(3),range(2),range(4)):
                if letters=='HS':
                    key = ('I' if i+j+k+l==0 else f"\omega^{i+3*j}"+'HSSS'*j+'HSSH'*k+'S'*l)
                if letters=='SEX':
                    key = ('I' if i+j+k+l==0 else 'E'*j+'X'*k+'S'*l)
                C1[key] = (omega**i)*np.linalg.matrix_power(E,j)@np.linalg.matrix_power(X,k)@np.linalg.matrix_power(S,l)
        return C1

    if nqubits == 2:
        if ignore_global_phase:
            from scipy.linalg import expm
            # RS = np.array(expm(-1j*np.pi*(gates.X(0).matrix()+gates.Y(0).matrix()+gates.Z(0).matrix())/np.sqrt(27.)))
            # fancy_display(RS,'RS')
            # fancy_display(-E,'-E')
            RS = -E 
            CNOT = gates.CNOT(0,1).matrix()
            iSWAP = gates.iSWAP(0,1).matrix()
            SWAP = gates.SWAP(0,1).matrix()
            
            C1 = clifford_group(1, ignore_global_phase,letters)
            if letters=='HS':
                S1 = {'I': np.eye(2), f'-\omega^{3}HSSS': RS, f'\omega^{6}HSSSHSSS': RS@RS}
            if letters=='SEX':
                S1 = {'I': np.eye(2), '-E': RS, 'EE': RS@RS}                
            C2 = {}
            for k1,k2 in itertools.product(C1.keys(),C1.keys()):
                kron = np.kron(C1[k1],C1[k2])
                C2['('+k1+'\otimes '+k2+')'] = kron
                for k3,k4 in itertools.product(S1.keys(),S1.keys()):
                    C2['('+k1+'\otimes '+k2+').CNOT.('+k3+'\otimes '+k4+')'] = kron@CNOT@np.kron(S1[k3],S1[k4])
                    C2['('+k1+'\otimes '+k2+').iSWAP.('+k3+'\otimes '+k4+')'] = kron@iSWAP@np.kron(S1[k3],S1[k4])
                C2['('+k1+'\otimes '+k2+').SWAP'] = kron@SWAP
            return C2
        else: 
            print("Not implemented")
            return None



def random_clifford(nqubits,ignore_global_phase=True,letters='HS'):
    
    """ Samples one element from the list of all Cliffords """
    
    C = clifford_group(nqubits,ignore_global_phase,letters)
    k = np.random.choice(list(C.keys()))
    return k, C[k] 
    
    

def random_U_fixed_T(nqubits, nT, ignore_global_phase=True, letters='HS'):    

    I = np.eye(2)
    T = gates.T(0).matrix()
    T_dict= {'I': I, 'T': T}

    if nqubits == 1:

        if nT == 0:
            c1k, c1u = random_clifford(nqubits,ignore_global_phase,letters)
            return c1k, c1u
        
        if nT > 0:
            c2k, c2u = random_clifford(nqubits,ignore_global_phase,letters)
            return c2k+'.T.'+random_U_fixed_T(nqubits, nT-1, ignore_global_phase, letters)[0], c2u@T@random_U_fixed_T(nqubits, nT-1, ignore_global_phase, letters)[1]

    if nqubits == 2:
        if nT == 0:
            c1k, c1u = random_clifford(nqubits,ignore_global_phase,letters)
            return c1k, c1u

        if nT == 1:
            c2k, c2u = random_clifford(nqubits,ignore_global_phase,letters)
            Tcount = 0
            while Tcount != 1: # making sure we sample exactly one T to add 
                ks = np.random.choice(['I','T'],2)
                Tcount = (ks[0]+ks[1]).count('T')
            TT = np.kron(T_dict[ks[0]],T_dict[ks[1]])
            return c2k+f'.{ks[0]} \otimes {ks[1]}.'+random_U_fixed_T(nqubits, nT-Tcount, ignore_global_phase, letters)[0], c2u@TT@random_U_fixed_T(nqubits, nT-Tcount, ignore_global_phase, letters)[1]

        if nT > 1:
            c2k, c2u = random_clifford(nqubits,ignore_global_phase,letters)
            Tcount = 0
            while Tcount < 1: # making sure we sample at least one T to add 
                ks = np.random.choice(['I','T'],2)
                Tcount += (ks[0]+ks[1]).count('T')
            TT = np.kron(T_dict[ks[0]],T_dict[ks[1]])
            return c2k+f'.{ks[0]} \otimes {ks[1]}.'+random_U_fixed_T(nqubits, nT-Tcount, ignore_global_phase, letters)[0], c2u@TT@random_U_fixed_T(nqubits, nT-Tcount, ignore_global_phase, letters)[1]
                


############################################## NOISY ###############################################


def apply_noise_to_channel(noise,channel):  

    """
    Apply noise to a channel
    Input: noise object from qibo, channel in Choi rep
    Output: noisy channel in Choi form
    """
    return qi.liouville_to_choi(noise.to_liouville()@qi.choi_to_liouville(channel))
    
    
def apply_noise_to_basis(B,noise_model):

    """ Constructs a noisy version of a basis B and checks if it's still a basis """

    # Noisy basis
    B_noisy = {k:apply_noise_to_channel(noise_model[k],B[k]) for k in B.keys()}
    dsquared = len(noise_model[list(B.keys())[0]].to_liouville()) # useful for the span check below when B is an overcomplete basis
    print(f'Applied noise model to basis elements')

    # Gram matrix and LI check
    G = gram_matrix(list(B_noisy.values()))
    rank = np.linalg.matrix_rank(G)
    if rank == dsquared**2-dsquared+1:
        print(f'The noisy channels form a basis! :)')
    else:
        print(f"No longer a basis! :(\nOnly spanned {rank} directions")   
    
    return B_noisy


def params_basic_2q(params_basic):

    """ 
    Input: depolarizing parameters for H, S, T, CX, Px, Py, Pz
    Output: depolarizing parameters for CX and tensor products of [I,H,S,T] and of [I,Px,Py,Pz]
    """
    params = {}
    
    for gate in set(['CX']):
        params[gate] = params_basic[gate]
    for gate in itertools.product(['I','H','S','T'],['I','H','S','T']):
        params[gate[0]+gate[1]] = params_basic[gate[0]]+params_basic[gate[1]]
    for gate in itertools.product(['I','Px','Py','Pz'],['I','Px','Py','Pz']):
        params[gate[0]+gate[1]] = params_basic[gate[0]]+params_basic[gate[1]]    
        
    return params     