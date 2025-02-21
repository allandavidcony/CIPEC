import numpy as np
import matplotlib.pyplot as plt

from qibo import quantum_info as qi
from qibo import gates, set_backend
from qibo.backends import NumpyBackend
from qibo.noise import NoiseModel, DepolarizingError
from qibo.quantum_info.metrics import diamond_norm

set_backend("numpy")

import itertools
from tqdm.notebook import tqdm

import os.path
import csv

##############################################################

def gram_matrix(B):
    
    """ naive Gram matrix """
    
    d = len(B)
    G = 1.j*np.zeros([d,d])
    for ii in range(d):
        for jj in range(ii+1,d):
            G[ii,jj] = np.trace(B[ii].conj().T@B[jj])
    G = (G+G.conj().T)
    for ii in range(d):
        G[ii,ii] = np.trace(B[ii].conj().T@B[ii])
    return G

def update_gram_matrix(oldG, B_prev, new_channel):
    
    """ more efficient Gram matrix that only adds a new row/column to an old one """
    
    d = len(oldG)
    G = np.zeros((d+1,d+1), dtype='complex_')
    G[:-1,:-1] = oldG
    for ii in range(d):
        G[ii,-1] = np.trace(B_prev[ii].conj().T@new_channel, dtype='complex_')
        G[-1,ii] = G[ii,-1].conj()
    G[-1,-1] = np.trace(new_channel.conj().T@new_channel, dtype='complex_')
    return G

def clifford_T_channels(n_qubits, include_T=True):

    """ returns dict of unitary channels in Choi representation """

    # 1q gates
    I = np.eye(2)
    H = (1.0/np.sqrt(2.0))*np.array(np.array([[1.,  1.],[1., -1.]]))
    S = np.array(np.array([[1., 0.],[0., 1.j]]))
    T = np.array(np.array([[1., 0.],[0., np.exp(1j*np.pi/4.)]]))

    # 2q gates
    II = np.eye(2**2)
    HI = np.kron(H,I)
    IH = np.kron(I,H)
    SI = np.kron(S,I)
    IS = np.kron(I,S)
    TI = np.kron(T,I)
    IT = np.kron(I,T)
    HS = np.kron(H,S)
    SH = np.kron(S,H)
    HT = np.kron(H,T)
    TH = np.kron(T,H)
    TS = np.kron(T,S)
    ST = np.kron(S,T)
    CX = np.array([[1., 0., 0., 0.],[0., 1., 0., 0.],[0., 0., 0., 1.],[0., 0., 1., 0.]])
    XC = np.array([[1., 0., 0., 0.],[0., 0., 0., 1.],[0., 0., 1., 0.],[0., 1., 0., 0.]])

    # corresponding channels in Choi rep.
    if n_qubits == 1:
        unitary_gates = {"H": H, "S": S}
        if include_T:
            unitary_gates["T"] = T

    if n_qubits == 2:
        unitary_gates = {"HI": HI, "IH": IH, "SI": SI, "IS": IS, "HS": HS, "SH": SH, "CX": CX, "XC": XC}
        if include_T:
            unitary_gates["TI"] = TI
            unitary_gates["IT"] = IT
            unitary_gates["HT"] = HT
            unitary_gates["TH"] = TH
            unitary_gates["ST"] = ST
            unitary_gates["TS"] = TS
#         unitary_gates = {"HI": HI, "IH": IH, "SI": SI, "IS": IS, "TI": TI, "IT": IT,"HS": HS,
#                          "HT": HT, "SH": SH, "ST": ST, "TS": TS, "TH": TH, "CX": CX, "XC": XC}

    unitary_channels = {k:qi.to_choi(v) for k,v in unitary_gates.items()}

    return unitary_channels

def state_prep_channels(n_qubits):

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
    if n_qubits == 1:
        channels = {"Px": np.kron(projector_0x, I), "Py": np.kron(projector_0y, I), "Pz": np.kron(projector_0z, I)}

    if n_qubits == 2:
        channels = {"PxI": np.kron(projector_0xI, II), "PyI": np.kron(projector_0yI, II), "PzI": np.kron(projector_0zI, II),
                    "IPx": np.kron(Iprojector_0x, II), "IPy": np.kron(Iprojector_0y, II), "IPz": np.kron(Iprojector_0z, II),
                    "PxPx": np.kron(projector_0x0x, II), "PxPy": np.kron(projector_0x0y, II), "PxPz": np.kron(projector_0x0z, II),
                    "PyPx": np.kron(projector_0y0x, II), "PyPy": np.kron(projector_0y0y, II), "PyPz": np.kron(projector_0y0z, II),
                    "PzPx": np.kron(projector_0z0x, II), "PzPy": np.kron(projector_0z0y, II), "PzPz": np.kron(projector_0z0z, II)}

    return channels



def write(data,filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)
        

def noiseless_basis(nqubits, include_T=True):
    
    """ builds a minimal basis by selecting LI sequences of unitary channels, then adding state prep channels """

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
        print('Loaded pre-computed basis')

    # if not then let's compute
    else:
        unitary_channels = clifford_T_channels(nqubits,include_T)
        nonunitary_channels = state_prep_channels(nqubits)

        CPTP_dim = 2**(4*nqubits) - 2**(2*nqubits) + 1
        print(f'dim(CPTP)={CPTP_dim}\n')

        # start composing only unitary channels
        keys = list(unitary_channels.keys())
        composite_keys = []
        B = {}
        rank = 0

        # words of length 1
        for k in list(itertools.product(keys)):
            composite_keys.append(tuple([k[0]]))
        # words of length 2
        for k in list(itertools.product(keys,keys)):
            composite_keys.append((k[0],k[1]))
        # words of length 3
        for k in list(itertools.product(keys,keys,keys)):
            composite_keys.append((k[0],k[1],k[2]))
        # words of length 4
        for k in list(itertools.product(keys,keys,keys,keys)):
            composite_keys.append((k[0],k[1],k[2],k[3]))

        print(f"Starting sequences of unitaries. There are {len(composite_keys)} candidate words up to L=4 to try.")

        composite_keys = tqdm(composite_keys)
        G = np.empty([0,0], dtype='complex_')
        for k in composite_keys:
            new_channel = qi.choi_to_liouville(unitary_channels[k[0]])
            if len(k) > 1:
                for i in range(1,len(k)):
                    new_channel = qi.choi_to_liouville(unitary_channels[k[i]])@new_channel
            new_channel = qi.liouville_to_choi(new_channel)
            GU = update_gram_matrix(G, list(B.values()), new_channel)
            if np.linalg.matrix_rank(GU) > rank:
                rank += 1
                B[k] = new_channel
                G = GU
                if rank == CPTP_dim-len(nonunitary_channels):
                    print("Success!")
                    break
            else:
                pass
            composite_keys.set_description(f'Current rank(G) = {rank}')

        print(f"Achieved rank(G)={rank}\n")

        # now add state prep channels
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
		
	
def apply_noise(B, noise):    
	return {k:qi.liouville_to_choi(noise.to_liouville()@qi.choi_to_liouville(v)) for k,v in B.items()}