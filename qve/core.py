import os
os.environ["CUQUANTUM_LOG_LEVEL"] = "OFF"
import os
import time
from itertools import  chain

import numpy as np

from cuquantum import *
import cupy as cp
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

from functools import cache


def make_bsp(n_dim):
    """Creates a basic quantum circuit for feature mapping."""
    param = ParameterVector("p", n_dim)
    bsp_qc = QuantumCircuit(n_dim)
    bsp_qc.h(list(range(n_dim)))
    for q in range(n_dim):
        bsp_qc.rz(param.params[q], [q])
        bsp_qc.ry(param.params[q], [q])
    for i in range(n_dim - 1):
        bsp_qc.cx(i, i + 1)
    for q in range(n_dim):
        bsp_qc.rz(param.params[q], [q])
    return bsp_qc

def build_qsvm_qc(bsp_qc, n_dim, y_t, x_t):
    """Builds a quantum circuit for the quantum SVM kernel."""
    qc_1 = bsp_qc.assign_parameters(y_t).to_gate()
    qc_2 = bsp_qc.assign_parameters(x_t).inverse().to_gate()
    kernel_qc = QuantumCircuit(n_dim)
    kernel_qc.append(qc_1, list(range(n_dim)))
    kernel_qc.append(qc_2, list(range(n_dim)))
    return kernel_qc

# EDIT V2
@cache
def sin_cos(d):
    return np.sin(d), np.cos(d)

# EDIT V2
@cache
def get_from_d1(d1):
    half_d1 = d1 / 2
    z_g  = [[np.exp(-1j*half_d1), 0],[0, np.exp(1j*half_d1)]]
    s, c = sin_cos(half_d1)
    return s, c, z_g
# EDIT V2
@cache
def get_from_d2(d2):
    half_d2 = d2 / 2
    z_gd  = [[np.exp(1j*half_d2),0],[0,np.exp(-1j*half_d2)]]
    s, c = sin_cos(half_d2)
    return s, c, z_gd

# EDIT V2: pre-computing parts..

def renew_operand(n_dim, oper_tmp, y_t, x_t):
    oper = oper_tmp.copy()
    n_zg, n_zy_g = [], []
    for d1 in y_t:
        s, c, z_g = get_from_d1(d1)
        n_zg.append(z_g)
        n_zy_g.extend([z_g, [[c, -s],[s, c]]])

    oper[n_dim*2:n_dim*4] = cp.array(n_zy_g)
    oper[n_dim*5-1:n_dim*6-1] = cp.array(n_zg)

    n_zgd, n_zy_gd = [], []
    for d2 in x_t[::-1]:
        s, c, z_gd = get_from_d2(d2)
        n_zgd.append(z_gd)
        n_zy_gd.extend([[[c, s],[-s, c]], z_gd])


    oper[n_dim*6-1:n_dim*7-1] = cp.array(n_zgd)
    oper[n_dim*8-2:n_dim*10-2] = cp.array(n_zy_gd)

    return oper

def data_partition(indices_list, size, rank):
    """
    Partitions a list of indices for distributed processing.
    """
    num_data = len(indices_list)
    chunk, extra = num_data // size, num_data % size
    data_begin = rank * chunk + min(rank, extra)
    data_end = num_data if rank == size - 1 else (rank + 1) * chunk + min(rank + 1, extra)
    return indices_list[data_begin:data_end]

# EDITED: reduced for-loop so its a comprenhension list 
def data_to_operand(n_dim,operand_tmp,data1,data2,indices_list):
    return [renew_operand(n_dim, operand_tmp, data1[i1-1], data2[i2-1]) for i1, i2 in indices_list]


# EDITED: removing the append so instead does direct mapping 
def operand_to_amp(opers, network):
    amp_tmp = [None]*len(opers)
    with network as tn:
        for i, op in enumerate(opers):
            tn.reset_operands(*op)
            amp_tmp[i] = abs(tn.contract()) ** 2
    return amp_tmp

def get_kernel_matrix(data1, data2, amp_data, indices_list, mode=None):
    """Builds the precomputed kernel matrix from amplitudes."""
    amp_m = list(chain.from_iterable(amp_data))
    kernel_matrix = np.zeros((len(data1), len(data2)))
    i = -1
    for i1, i2 in indices_list:
        i += 1
        kernel_matrix[i1 - 1][i2 - 1] = np.round(amp_m[i], 8)
    if mode == "train":
        kernel_matrix = kernel_matrix + kernel_matrix.T + np.diag(np.ones((len(data2))))
    return kernel_matrix

