#!/usr/bin/env python3
"""Common parts of TRG and HOTRG algorithms for the Ising model on the square lattice"""

import numpy as np
import scipy.linalg as spl
from ncon import ncon

Tc_O1Cub = np.sqrt(2)-1  # Critical temperature of the 2D cublc O(1) model on the square lattice.
Tc_O2Cub = 0.5  # Critical temperature of the 2D cubic O(2) model on the square lattice
Tc_O2Cub_rev = 1.0
Tc_O2Cub_rev_2 = 2.461
Tc_XY = 0.893  # BKT transition temperature of the 2D XY model on the square lattice.
Xc_O1SL = np.tanh(np.arcsinh(1)/2)
Xc_O2SL = 1.040


def svd(a, axes0, axes1, rank=None):
    """Singular value decomposition for tensor.

    Args:
        a: A tensor to be decomposed.
        axes0: Axes which connect to U.
        axes1: Axes which connect to VT.
        rank: The maximum number of singular values to be returned.
            If it is None (default), there is no truncation.

    Returns:
        U: A isometric tensor containing the left singular vectors.
            The last index corresponds to the singular values.
        S: The singular values.
        VT: A isometric tensor containing the right singular vectors.
            The first index corresponds to the singular values.

    Note:
        The function calculates the full SVD and then truncates it.
        Therefore, it is not expected to reduce the computational cost.
    """

    shape = np.array(a.shape)
    shape_row = [shape[i] for i in axes0]
    shape_col = [shape[i] for i in axes1]
    
    n_row = np.prod(shape_row)
    n_col = np.prod(shape_col)

    chi = min(n_row, n_col)
    if (rank is not None) and (rank < chi):
        chi = rank

    mat = np.reshape(np.transpose(a, axes0 + axes1), (n_row, n_col))
    u, s, vt = spl.svd(mat, full_matrices=False)

    if chi < len(s):
        u = u[:, 0:chi]
        s = s[0:chi]
        vt = vt[0:chi, :]

    n = len(s)
    return u.reshape(shape_row + [n]), s, vt.reshape([n] + shape_col)


def initial_TN(temp):
    """Initial tensor of the ising model on the square lattice.

    Args:
        temp: Temperature

    Returns:
        a: Initial 4-leg tensor. [top, right, bottom, left]
        log_factor: Logarithm of the normalization factor.
        n_spin: The number of spins which contained the initial tensor.
    """

    shape = (2, 2, 2, 2)
    a = np.zeros(shape, dtype=float)  # [top, right, bottom, left]
    c = np.cosh(1.0 / temp)
    s = np.sinh(1.0 / temp)
    
    for idx in np.ndindex(shape):
        if sum(idx) == 0:
            a[idx] = 2 * c * c
        elif sum(idx) == 2:
            a[idx] = 2 * c * s
        elif sum(idx) == 4:
            a[idx] = 2 * s * s

    # normalize
    val = np.einsum("ijij", a)
    a /= val
    log_factor = np.log(val)

    n_spin = 1.0  # An initial tensor has one spin.
    return (a, log_factor, n_spin)

def initial_TNO1HL(temp):
    beta = 1/temp

    

    shape = (2, 2, 2) #[top, right, left] in XhaoXie2016
    T = np.zeros(shape, dtype = float)

    
    T[0, 0, 0] = np.exp(beta * 3/2)
    T[0, 1, 1] = np.exp(-beta/2)
    T[1, 0, 1] = np.exp(-beta/2)
    T[1, 1, 0] = np.exp(-beta/2)

    a = ncon([T, T], [[-1, 1, -4], [-2, -3, 1]])

    # normalize
    val = np.einsum("ijij", a)
    a /= val
    log_factor = np.log(val)

    n_spin = 1.0  # An initial tensor has one spin.
    return (a, log_factor, n_spin)

def initial_TNforB(temp):
    """Initial tensor of the ising model on the square lattice.

    Args:
        temp: Temperature

    Returns:
        a: Initial 4-leg tensor. [top, right, bottom, left]
        log_factor: Logarithm of the normalization factor.
        n_spin: The number of spins which contained the initial tensor.
    """

    shape = (2, 2, 2, 2)
    a = np.zeros(shape, dtype=float)  # [top, right, bottom, left]
    c = np.cosh(1.0 / temp)
    s = np.sinh(1.0 / temp)
    
    
    for idx in np.ndindex(shape):
        if sum(idx) == 0:
            a[idx] = 2 * c * c
        elif sum(idx) == 2:
            a[idx] = 2 * c * s
        elif sum(idx) == 4:
            a[idx] = 2 * s * s
    
    '''
    a = np.ones(shape, dtype=float)  # [top, right, bottom, left]
    a[0, 1, 0, 1] = a[1, 0, 1, 0] = np.exp(-4.0 / temp)
    a[0, 0, 0, 0] = a[1, 1, 1, 1] = np.exp(4.0 / temp)
    '''

    n_spin = 1.0  # An initial tensor has one spin.

    shape2 = (2, 2)
    s = np.zeros(shape2, dtype=float)
    s[0, 0] = s[1, 1] = 1

    return (a, s, n_spin)

def initial_TN_O1S(theta):

    shape = (3, 3, 3, 3)
    T = np.zeros(shape, dtype=float)  # [top, right, bottom, left]

    t = theta

    w = 1/ (2 - (1 - 2 * np.sin(t/2)) * ((1 + 2 * np.sin(t/2))**2))
    u = 4 * w * np.sin(t/2) * np.cos(np.pi / 4 - t /4)
    v = w * (1 + 2 * np.sin(t/2))
 

    
    T[0, 0, 0, 0] = 1
    T[1, 1, 0, 0] = T[0, 1, 1, 0] = T[0, 0, 1, 1] = T[1, 0, 0, 1] = u
    T[1, 0, 1, 0] = T[0, 1, 0, 1] = v
    T[1, 1, 1, 1] = T[2, 2, 2, 2] = 2 * w

    # normalize
    val = np.einsum("ijij", T)
    T /= val
    log_factor = np.log(val)

    n_spin = 1.0  # An initial tensor has one spin.
    return (T, log_factor, n_spin)

def initial_TN_O1SforBrev(theta):

    shape = (3, 3, 3, 3)
    T = np.zeros(shape, dtype=float)  # [top, right, bottom, left]

    t = theta

    w = 1/ (2 - (1 - 2 * np.sin(t/2)) * ((1 + 2 * np.sin(t/2))**2))
    u = 4 * w * np.sin(t/2) * np.cos(np.pi / 4 - t /4)
    v = w * (1 + 2 * np.sin(t/2))
 

    
    T[0, 0, 0, 0] = 1
    T[0, 1, 1, 0] = T[1, 0, 0, 1] = u
    T[2, 2, 0, 0] = T[0, 0, 2, 2] = u

    T[1, 0, 1, 0] = T[0, 1, 0, 1] = v
    T[2, 0, 2, 0] = T[0, 2, 0, 2] = v
    T[1, 0, 2, 0] = T[2, 0, 1, 0] = v
    T[0, 1, 0, 2] = T[0, 2, 0, 1] = v

    T[1, 1, 1, 1] = T[2, 2, 2, 2] = w

    # normalize
    val = np.einsum("ijij", T)
    T /= val
    log_factor = np.log(val)

    shape2 = (3, 3)
    s = np.zeros(shape2, dtype=float)
    s[0, 0] = s[1, 1] = s[2, 2] = 1
    s = s / 3

    n_spin = 1.0  # An initial tensor has one spin.
    return (T, s, log_factor, n_spin)

def initial_TN_O1SforB(theta):

    shape = (2, 2, 2, 2)
    T = np.zeros(shape, dtype=float)  # [top, right, bottom, left]

    t = theta

    w = 1/ (2 - (1 - 2 * np.sin(t/2)) * ((1 + 2 * np.sin(t/2))**2))
    u = 4 * w * np.sin(t/2) * np.cos(np.pi / 4 - t /4)
    v = w * (1 + 2 * np.sin(t/2))
 

    
    T[0, 0, 0, 0] = 1
    T[1, 1, 0, 0] = T[0, 1, 1, 0] = T[0, 0, 1, 1] = T[1, 0, 0, 1] = u
    T[1, 0, 1, 0] = T[0, 1, 0, 1] = v
    T[1, 1, 1, 1] = 2*w

    # normalize
    val = np.einsum("ijij", T)
    T /= val
    log_factor = np.log(val)

    shape2 = (2, 2)
    s = np.zeros(shape2, dtype=float)
    s[0, 0] = s[1, 1] = 1
    #s = s / 2

    n_spin = 1.0  # An initial tensor has one spin.
    return (T, s, log_factor, n_spin)

def initial_TN_O1SFPLforB():

    shape = (2, 2, 2, 2)
    T = np.zeros(shape, dtype=float)  # [top, right, bottom, left]

    '''
    t = theta

    w = 1/ (2 - (1 - 2 * np.sin(t/2)) * ((1 + 2 * np.sin(t/2))**2))
    u = 4 * w * np.sin(t/2) * np.cos(np.pi / 4 - t /4)
    v = w * (1 + 2 * np.sin(t/2))
    '''

    u = v = 1.0  # For FPL, we set all weights to 1.0.
    w = 0
 

    
    T[0, 0, 0, 0] = 0
    T[1, 1, 0, 0] = T[0, 1, 1, 0] = T[0, 0, 1, 1] = T[1, 0, 0, 1] = u
    T[1, 0, 1, 0] = T[0, 1, 0, 1] = v
    #T[1, 1, 1, 1] = T[2, 2, 2, 2] = 2*w

    # normalize
    val = np.einsum("ijij", T)
    T /= val
    log_factor = np.log(val)

    shape2 = (2, 2)
    s = np.zeros(shape2, dtype=float)
    s[0, 0] = s[1, 1] = 1
    s = s / 2

    n_spin = 1.0  # An initial tensor has one spin.
    return (T, s, log_factor, n_spin)

def initial_TN_O2SforB(theta):

    shape = (3, 3, 3, 3)
    T = np.zeros(shape, dtype=float)  # [top, right, bottom, left]

    t = theta

    w = 1/ (2 - (1 - 2 * np.sin(t/2)) * ((1 + 2 * np.sin(t/2))**2))
    u = 4 * w * np.sin(t/2) * np.cos(np.pi / 4 - t /4)
    v = w * (1 + 2 * np.sin(t/2))
 

    
    T[0, 0, 0, 0] = 1
    T[1, 1, 0, 0] = T[0, 1, 1, 0] = T[0, 0, 1, 1] = T[1, 0, 0, 1] = u
    T[2, 2, 0, 0] = T[0, 2, 2, 0] = T[0, 0, 2, 2] = T[2, 0, 0, 2] = u

    T[1, 0, 1, 0] = T[0, 1, 0, 1] = v
    T[2, 0, 2, 0] = T[0, 2, 0, 2] = v

    T[1, 1, 1, 1] = T[2, 2, 2, 2] = 2 * w

    T[1, 1, 2, 2] = T[2, 2, 1, 1] = T[1, 2, 2, 1] = T[2, 1, 1, 2] = w



    # normalize
    val = np.einsum("ijij", T)
    T /= val
    log_factor = np.log(val)

    shape2 = (3, 3)
    s = np.zeros(shape2, dtype=float)
    s[0, 0] = s[1, 1] = s[2, 2] = 1
    #s = s / 3

    n_spin = 1.0  # An initial tensor has one spin.
    return (T, s, log_factor, n_spin)

def initial_TN_O1Sb0forB(theta):

    shape = (2, 2, 2, 2)
    T = np.zeros(shape, dtype=float)  # [top, right, bottom, left]

    t = theta

    w = 1/ (2 - (1 - 2 * np.sin(t/2)) * ((1 + 2 * np.sin(t/2))**2))
    u = 4 * w * np.sin(t/2) * np.cos(np.pi / 4 - t /4)
    v = w * (1 + 2 * np.sin(t/2))
 

    
    T[0, 0, 0, 0] = 1
    T[1, 1, 0, 0] = T[0, 1, 1, 0] = T[0, 0, 1, 1] = T[1, 0, 0, 1] = 1/2
    T[1, 0, 1, 0] = T[0, 1, 0, 1] = 0
    T[1, 1, 1, 1] = 2 * (1/2)

    # normalize
    val = np.einsum("ijij", T)
    T /= val
    log_factor = np.log(val)

    shape2 = (2, 2)
    s = np.zeros(shape2, dtype=float)
    s[0, 0] = s[1, 1] = 1
    s = s / 2

    n_spin = 1.0  # An initial tensor has one spin.
    return (T, s, log_factor, n_spin)

def initial_TN_O1SLforB(temp):
    # Z = \sum x^{length} n^{# of loops}

    shape = (2, 2, 2, 2)
    T = np.zeros(shape, dtype=float)  # [top, right, bottom, left]

    x = 1.0 / temp
    y = x**2
 

    
    T[0, 0, 0, 0] = 1
    T[1, 1, 0, 0] = T[0, 1, 1, 0] = T[0, 0, 1, 1] = T[1, 0, 0, 1] = x
    T[1, 0, 1, 0] = T[0, 1, 0, 1] = x
    T[1, 1, 1, 1] = 0

    # normalize
    val = np.einsum("ijij", T)
    T /= val
    log_factor = np.log(val)

    shape2 = (2, 2)
    s = np.zeros(shape2, dtype=float)
    s[0, 0] = s[1, 1] = 1
    

    n_spin = 1.0  # An initial tensor has one spin.
    return (T, s, log_factor, n_spin)

def initial_TN_O1SLa(temp):
    # Z = \sum x^{length} n^{# of loops}

    shape = (2, 2, 2, 2)
    T = np.zeros(shape, dtype=float)  # [top, right, bottom, left]

    x = temp
    y = x**2
 

    
    T[0, 0, 0, 0] = 1
    T[1, 1, 0, 0] = T[0, 1, 1, 0] = T[0, 0, 1, 1] = T[1, 0, 0, 1] = x
    T[1, 0, 1, 0] = T[0, 1, 0, 1] = x
    T[1, 1, 1, 1] = y

    # normalize
    val = np.einsum("ijij", T)
    T /= val
    log_factor = np.log(val)
    

    n_spin = 1.0  # An initial tensor has one spin.
    return (T, log_factor, n_spin)

def initial_TN_O1SLaforB(temp):
    # Z = \sum x^{length} n^{# of loops}

    shape = (2, 2, 2, 2)
    T = np.zeros(shape, dtype=float)  # [top, right, bottom, left]

    x = temp
    y = x**2
 

    
    T[0, 0, 0, 0] = 1
    T[1, 1, 0, 0] = T[0, 1, 1, 0] = T[0, 0, 1, 1] = T[1, 0, 0, 1] = x
    T[1, 0, 1, 0] = T[0, 1, 0, 1] = x
    T[1, 1, 1, 1] = y

    # normalize
    val = np.einsum("ijij", T)
    T /= val
    log_factor = np.log(val)

    shape2 = (2, 2)
    s = np.zeros(shape2, dtype=float)
    s[0, 0] = s[1, 1] = 1
    

    n_spin = 1.0  # An initial tensor has one spin.
    return (T, s, log_factor, n_spin)

def initial_TN_O1CubforB(temp):
    # Z = \sum x^{length} n^{# of loops}

    shape = (2, 2, 2, 2)
    T = np.zeros(shape, dtype=float)  # [top, right, bottom, left]

    x = 1.0 / temp
    y = x**2
 

    
    T[0, 0, 0, 0] = 1
    T[1, 1, 0, 0] = T[0, 1, 1, 0] = T[0, 0, 1, 1] = T[1, 0, 0, 1] = x
    T[1, 0, 1, 0] = T[0, 1, 0, 1] = x
    T[1, 1, 1, 1] = x**2

    # normalize
    val = np.einsum("ijij", T)
    T /= val
    log_factor = np.log(val)

    shape2 = (2, 2)
    s = np.zeros(shape2, dtype=float)
    s[0, 0] = s[1, 1] = 1
    

    n_spin = 1.0  # An initial tensor has one spin.
    return (T, s, log_factor, n_spin)

def initial_TN_O1CubforB_rev(temp):
    # Z = \sum x^{length} n^{# of loops}

    shape = (2, 2, 2, 2)
    T = np.zeros(shape, dtype=float)  # [top, right, bottom, left]

    x = 1.0 / temp
    y = x**2
 

    
    T[0, 0, 0, 0] = 1
    T[1, 1, 0, 0] = T[0, 1, 1, 0] = T[0, 0, 1, 1] = T[1, 0, 0, 1] = x
    T[1, 0, 1, 0] = T[0, 1, 0, 1] = x
    T[1, 1, 1, 1] = x**2

    

    shape2 = (2, 2)
    s = np.zeros(shape2, dtype=float)
    s[0, 0] = s[1, 1] = 1
    

    n_spin = 1.0  # An initial tensor has one spin.
    return (T, s, n_spin)

def initial_TN_O2CubforB(temp):
    # Z = \sum x^{length} n^{# of loops}

    shape = (3, 3, 3, 3)
    T = np.zeros(shape, dtype=float)  # [top, right, bottom, left]

    x = temp
    y = x**2
 

    
    T[0, 0, 0, 0] = 1
    T[1, 1, 0, 0] = T[0, 1, 1, 0] = T[0, 0, 1, 1] = T[1, 0, 0, 1] = x
    T[1, 0, 1, 0] = T[0, 1, 0, 1] = x
    T[2, 2, 0, 0] = T[0, 2, 2, 0] = T[0, 0, 2, 2] = T[2, 0, 0, 2] = x
    T[2, 0, 2, 0] = T[0, 2, 0, 2] = x
    T[1, 1, 1, 1] = x**2
    T[2, 2, 2, 2] = x**2

    
    '''
    #If you allow to share two same sites
    T[1, 1, 2, 2] = x**2
    T[2, 1, 1, 2] = x**2
    T[2, 2, 1, 1] = x**2
    T[1, 2, 2, 1] = x**2
    '''
    
    '''
    T[1, 2, 1, 2] = x**2
    T[2, 1, 2, 1] = x**2
    '''

    '''
    T[1, 2, 0, 0] = x
    T[2, 1, 0, 0] = x
    T[0, 2, 1, 0] = x
    T[0, 1, 2, 0] = x
    T[0, 0, 1, 2] = x
    T[0, 0, 2, 1] = x
    T[1, 0, 0, 2] = x
    T[2, 0, 0, 1] = x
    T[1, 2, 2, 2] = x**2
    T[2, 1, 2, 2] = x**2
    T[2, 2, 1, 2] = x**2
    T[2, 2, 2, 1] = x**2
    T[2, 1, 1, 1] = x**2
    T[1, 2, 1, 1] = x**2
    T[1, 1, 2, 1] = x**2
    T[1, 1, 1, 2] = x**2
    '''

    
    

    

    # normalize
    '''
    val = np.einsum("ijij", T)
    T /= val
    log_factor = np.log(val)
    '''

    shape2 = (3, 3)
    s = np.zeros(shape2, dtype=float)
    s[0, 0] = s[1, 1] = s[2, 2] = 1
    

    n_spin = 1.0  # An initial tensor has one spin.
    return (T, s, n_spin)

def initial_TN_O2Cub_rev_forB(temp):
    # Z = \sum x^{length} n^{# of loops}

    shape = (3, 3, 3, 3)
    T = np.zeros(shape, dtype=float)  # [top, right, bottom, left]

    x = temp
    y = x**2

    T[0, 0, 0, 0] = 2
    T[1, 1, 0, 0] = T[0, 1, 1, 0] = T[0, 0, 1, 1] = T[1, 0, 0, 1] = x
    T[1, 0, 1, 0] = T[0, 1, 0, 1] = x
    T[2, 2, 0, 0] = T[0, 2, 2, 0] = T[0, 0, 2, 2] = T[2, 0, 0, 2] = x
    T[2, 0, 2, 0] = T[0, 2, 0, 2] = x
    T[1, 1, 1, 1] = x**2
    T[2, 2, 2, 2] = x**2

    shape2 = (3, 3)
    s = np.zeros(shape2, dtype=float)
    s[0, 0] = s[1, 1] = s[2, 2] = 1

    n_spin = 1.0  # An initial tensor has one spin.
    return (T, s, n_spin)

def initial_TN_OnCub_rev_forB(temp, n):
    """Initial tensor of the loop model on the square lattice for general n.

    Args:
        temp: Temperature (or coupling parameter)
        n: The parameter for the model (bond dimension = n + 1)

    Returns:
        T: Initial 4-leg tensor. [top, right, bottom, left], shape (n+1, n+1, n+1, n+1)
        s: Bond weight matrix, shape (n+1, n+1), diagonal with 1s
        n_spin: The number of spins which contained the initial tensor (always 1.0)
    """
    # Z = \sum x^{length} n^{# of loops}

    shape = (n + 1, n + 1, n + 1, n + 1)
    T = np.zeros(shape, dtype=float)  # [top, right, bottom, left]

    x = temp

    # All indices are 0: value is n
    T[0, 0, 0, 0] = n
    
    # 2 indices same (value i), 2 indices 0: value is x
    # 4 indices all same (value i): value is x^2
    for i in range(1, n + 1):
        # 2つが同じ値iで他2つが0のパターン
        T[i, i, 0, 0] = T[0, i, i, 0] = T[0, 0, i, i] = T[i, 0, 0, i] = x
        T[i, 0, i, 0] = T[0, i, 0, i] = x
        
        # 4つ全てがiのとき
        T[i, i, i, i] = x**2

    # Bond weight matrix: diagonal matrix with all 1s
    shape2 = (n + 1, n + 1)
    s = np.zeros(shape2, dtype=float)
    for i in range(n + 1):
        s[i, i] = 1

    n_spin = 1.0  # An initial tensor has one spin.
    return (T, s, n_spin)

def initial_TN_O2Cub_NoIV_forB(temp):
    # Z = \sum x^{length} n^{# of loops}

    shape = (3, 3, 3, 3)
    T = np.zeros(shape, dtype=float)  # [top, right, bottom, left]

    x = temp
    y = x**2

    T[0, 0, 0, 0] = 1
    T[1, 1, 0, 0] = T[0, 1, 1, 0] = T[0, 0, 1, 1] = T[1, 0, 0, 1] = x
    T[1, 0, 1, 0] = T[0, 1, 0, 1] = x
    T[2, 2, 0, 0] = T[0, 2, 2, 0] = T[0, 0, 2, 2] = T[2, 0, 0, 2] = x
    T[2, 0, 2, 0] = T[0, 2, 0, 2] = x
    T[1, 1, 1, 1] = x**2
    T[2, 2, 2, 2] = x**2

    shape2 = (3, 3)
    s = np.zeros(shape2, dtype=float)
    s[0, 0] = s[1, 1] = s[2, 2] = 1

    n_spin = 1.0  # An initial tensor has one spin.
    return (T, s, n_spin)



def initial_TN_O2CubShareforB(temp):
    # Z = \sum x^{length} n^{# of loops}

    shape = (3, 3, 3, 3)
    T = np.zeros(shape, dtype=float)  # [top, right, bottom, left]

    x = temp
    y = x**2
 

    
    T[0, 0, 0, 0] = 1
    T[1, 1, 0, 0] = T[0, 1, 1, 0] = T[0, 0, 1, 1] = T[1, 0, 0, 1] = x
    T[1, 0, 1, 0] = T[0, 1, 0, 1] = x
    T[2, 2, 0, 0] = T[0, 2, 2, 0] = T[0, 0, 2, 2] = T[2, 0, 0, 2] = x
    T[2, 0, 2, 0] = T[0, 2, 0, 2] = x
    T[1, 1, 1, 1] = 2*x**2
    T[2, 2, 2, 2] = 2*x**2

    
    
    #If you allow to share two same sites
    T[1, 1, 2, 2] = x**2
    T[2, 1, 1, 2] = x**2
    T[2, 2, 1, 1] = x**2
    T[1, 2, 2, 1] = x**2
    
    
    '''
    T[1, 2, 1, 2] = x**2
    T[2, 1, 2, 1] = x**2
    '''

    '''
    T[1, 2, 0, 0] = x
    T[2, 1, 0, 0] = x
    T[0, 2, 1, 0] = x
    T[0, 1, 2, 0] = x
    T[0, 0, 1, 2] = x
    T[0, 0, 2, 1] = x
    T[1, 0, 0, 2] = x
    T[2, 0, 0, 1] = x
    T[1, 2, 2, 2] = x**2
    T[2, 1, 2, 2] = x**2
    T[2, 2, 1, 2] = x**2
    T[2, 2, 2, 1] = x**2
    T[2, 1, 1, 1] = x**2
    T[1, 2, 1, 1] = x**2
    T[1, 1, 2, 1] = x**2
    T[1, 1, 1, 2] = x**2
    '''

    
    

    

    # normalize
    val = np.einsum("ijij", T)
    T /= val
    log_factor = np.log(val)

    shape2 = (3, 3)
    s = np.zeros(shape2, dtype=float)
    s[0, 0] = s[1, 1] = s[2, 2] = 1
    

    n_spin = 1.0  # An initial tensor has one spin.
    return (T, s, log_factor, n_spin)

def initial_TN_O2CubCrossingforB(temp):
    # Z = \sum x^{length} n^{# of loops}

    shape = (3, 3, 3, 3)
    T = np.zeros(shape, dtype=float)  # [top, right, bottom, left]

    x = temp
    y = x**2
 

    
    T[0, 0, 0, 0] = 1
    T[1, 1, 0, 0] = T[0, 1, 1, 0] = T[0, 0, 1, 1] = T[1, 0, 0, 1] = x
    T[1, 0, 1, 0] = T[0, 1, 0, 1] = x
    T[2, 2, 0, 0] = T[0, 2, 2, 0] = T[0, 0, 2, 2] = T[2, 0, 0, 2] = x
    T[2, 0, 2, 0] = T[0, 2, 0, 2] = x
    T[1, 1, 1, 1] = x**2
    T[2, 2, 2, 2] = x**2

    
    
    #If you allow to share two same sites
    T[1, 1, 2, 2] = x**2
    T[2, 1, 1, 2] = x**2
    T[2, 2, 1, 1] = x**2
    T[1, 2, 2, 1] = x**2
    
    
    
    T[1, 2, 1, 2] = x**2
    T[2, 1, 2, 1] = x**2
    

    '''
    T[1, 2, 0, 0] = x
    T[2, 1, 0, 0] = x
    T[0, 2, 1, 0] = x
    T[0, 1, 2, 0] = x
    T[0, 0, 1, 2] = x
    T[0, 0, 2, 1] = x
    T[1, 0, 0, 2] = x
    T[2, 0, 0, 1] = x
    T[1, 2, 2, 2] = x**2
    T[2, 1, 2, 2] = x**2
    T[2, 2, 1, 2] = x**2
    T[2, 2, 2, 1] = x**2
    T[2, 1, 1, 1] = x**2
    T[1, 2, 1, 1] = x**2
    T[1, 1, 2, 1] = x**2
    T[1, 1, 1, 2] = x**2
    '''

    
    

    

    # normalize
    val = np.einsum("ijij", T)
    T /= val
    log_factor = np.log(val)

    shape2 = (3, 3)
    s = np.zeros(shape2, dtype=float)
    s[0, 0] = s[1, 1] = s[2, 2] = 1
    

    n_spin = 1.0  # An initial tensor has one spin.
    return (T, s, log_factor, n_spin)

def initial_TN_O2SLforB(temp):
    # Z = \sum x^{length} n^{# of loops}

    shape = (3, 3, 3, 3)
    T = np.zeros(shape, dtype=float)  # [top, right, bottom, left]

    x = temp
    y = x**2
 

    
    T[0, 0, 0, 0] = 1
    T[1, 1, 0, 0] = T[0, 1, 1, 0] = T[0, 0, 1, 1] = T[1, 0, 0, 1] = x
    T[1, 0, 1, 0] = T[0, 1, 0, 1] = x
    T[2, 2, 0, 0] = T[0, 2, 2, 0] = T[0, 0, 2, 2] = T[2, 0, 0, 2] = x
    T[2, 0, 2, 0] = T[0, 2, 0, 2] = x
    T[1, 1, 1, 1] = 3*x**2
    T[2, 2, 2, 2] = 3*x**2

    
    
    #If you allow to share two same sites
    T[1, 1, 2, 2] = x**2
    T[2, 1, 1, 2] = x**2
    T[2, 2, 1, 1] = x**2
    T[1, 2, 2, 1] = x**2
    
    
    
    T[1, 2, 1, 2] = x**2
    T[2, 1, 2, 1] = x**2
    

    '''
    T[1, 2, 0, 0] = x
    T[2, 1, 0, 0] = x
    T[0, 2, 1, 0] = x
    T[0, 1, 2, 0] = x
    T[0, 0, 1, 2] = x
    T[0, 0, 2, 1] = x
    T[1, 0, 0, 2] = x
    T[2, 0, 0, 1] = x
    T[1, 2, 2, 2] = x**2
    T[2, 1, 2, 2] = x**2
    T[2, 2, 1, 2] = x**2
    T[2, 2, 2, 1] = x**2
    T[2, 1, 1, 1] = x**2
    T[1, 2, 1, 1] = x**2
    T[1, 1, 2, 1] = x**2
    T[1, 1, 1, 2] = x**2
    '''

    
    

    

    # normalize
    val = np.einsum("ijij", T)
    T /= val
    log_factor = np.log(val)

    shape2 = (3, 3)
    s = np.zeros(shape2, dtype=float)
    s[0, 0] = s[1, 1] = s[2, 2] = 1
    

    n_spin = 1.0  # An initial tensor has one spin.
    return (T, s, log_factor, n_spin)

def initial_TN_O2SLaforB(temp):
    # Z = \sum x^{length} n^{# of loops}

    shape = (3, 3, 3, 3)
    T = np.zeros(shape, dtype=float)  # [top, right, bottom, left]

    x = temp /2
    y = temp**2 / 8
 

    
    T[0, 0, 0, 0] = 1
    T[1, 1, 0, 0] = T[0, 1, 1, 0] = T[0, 0, 1, 1] = T[1, 0, 0, 1] = x
    T[1, 0, 1, 0] = T[0, 1, 0, 1] = x
    T[2, 2, 0, 0] = T[0, 2, 2, 0] = T[0, 0, 2, 2] = T[2, 0, 0, 2] = x
    T[2, 0, 2, 0] = T[0, 2, 0, 2] = x
    T[1, 1, 1, 1] = 3*y
    T[2, 2, 2, 2] = 3*y

    
    
    #If you allow to share two same sites
    T[1, 1, 2, 2] = y
    T[2, 1, 1, 2] = y
    T[2, 2, 1, 1] = y
    T[1, 2, 2, 1] = y
    T[1, 2, 1, 2] = y
    T[2, 1, 2, 1] = y
    

    '''
    T[1, 2, 0, 0] = x
    T[2, 1, 0, 0] = x
    T[0, 2, 1, 0] = x
    T[0, 1, 2, 0] = x
    T[0, 0, 1, 2] = x
    T[0, 0, 2, 1] = x
    T[1, 0, 0, 2] = x
    T[2, 0, 0, 1] = x
    T[1, 2, 2, 2] = x**2
    T[2, 1, 2, 2] = x**2
    T[2, 2, 1, 2] = x**2
    T[2, 2, 2, 1] = x**2
    T[2, 1, 1, 1] = x**2
    T[1, 2, 1, 1] = x**2
    T[1, 1, 2, 1] = x**2
    T[1, 1, 1, 2] = x**2
    '''

    
    

    

    # normalize
    '''
    val = np.einsum("ijij", T)
    T /= val
    log_factor = np.log(val)
    '''

    shape2 = (3, 3)
    s = np.zeros(shape2, dtype=float)
    s[0, 0] = s[1, 1] = s[2, 2] = 1
    
    

    n_spin = 1.0  # An initial tensor has one spin.
    return (T, s, n_spin)

def initial_TN_O2SLa(temp):
    # Z = \sum x^{length} n^{# of loops}

    shape = (3, 3, 3, 3)
    T = np.zeros(shape, dtype=float)  # [top, right, bottom, left]

    x = temp /2
    y = temp**2 / 8
 

    
    T[0, 0, 0, 0] = 1
    T[1, 1, 0, 0] = T[0, 1, 1, 0] = T[0, 0, 1, 1] = T[1, 0, 0, 1] = x
    T[1, 0, 1, 0] = T[0, 1, 0, 1] = x
    T[2, 2, 0, 0] = T[0, 2, 2, 0] = T[0, 0, 2, 2] = T[2, 0, 0, 2] = x
    T[2, 0, 2, 0] = T[0, 2, 0, 2] = x
    T[1, 1, 1, 1] = 3*y
    T[2, 2, 2, 2] = 3*y

    
    
    #If you allow to share two same sites
    T[1, 1, 2, 2] = y
    T[2, 1, 1, 2] = y
    T[2, 2, 1, 1] = y
    T[1, 2, 2, 1] = y
    T[1, 2, 1, 2] = y
    T[2, 1, 2, 1] = y
    

    '''
    T[1, 2, 0, 0] = x
    T[2, 1, 0, 0] = x
    T[0, 2, 1, 0] = x
    T[0, 1, 2, 0] = x
    T[0, 0, 1, 2] = x
    T[0, 0, 2, 1] = x
    T[1, 0, 0, 2] = x
    T[2, 0, 0, 1] = x
    T[1, 2, 2, 2] = x**2
    T[2, 1, 2, 2] = x**2
    T[2, 2, 1, 2] = x**2
    T[2, 2, 2, 1] = x**2
    T[2, 1, 1, 1] = x**2
    T[1, 2, 1, 1] = x**2
    T[1, 1, 2, 1] = x**2
    T[1, 1, 1, 2] = x**2
    '''

    
    

    

    # normalize
    
    val = np.einsum("ijij", T)
    T /= val
    log_factor = np.log(val)
    
    

    n_spin = 1.0  # An initial tensor has one spin.
    return (T, log_factor, n_spin)

def initial_TN_O2Cub(temp):
    # Z = \sum x^{length} n^{# of loops}

    shape = (3, 3, 3, 3)
    T = np.zeros(shape, dtype=float)  # [top, right, bottom, left]

    x = temp
    y = x**2
 

    
    T[0, 0, 0, 0] = 1
    T[1, 1, 0, 0] = T[0, 1, 1, 0] = T[0, 0, 1, 1] = T[1, 0, 0, 1] = x
    T[1, 0, 1, 0] = T[0, 1, 0, 1] = x
    T[2, 2, 0, 0] = T[0, 2, 2, 0] = T[0, 0, 2, 2] = T[2, 0, 0, 2] = x
    T[2, 0, 2, 0] = T[0, 2, 0, 2] = x
    T[1, 1, 1, 1] = x**2
    T[2, 2, 2, 2] = x**2

    
    '''
    #If you allow to share two same sites
    T[1, 1, 2, 2] = x**2
    T[2, 1, 1, 2] = x**2
    T[2, 2, 1, 1] = x**2
    T[1, 2, 2, 1] = x**2
    '''
    
    '''
    T[1, 2, 1, 2] = x**2
    T[2, 1, 2, 1] = x**2
    '''

    '''
    T[1, 2, 0, 0] = x
    T[2, 1, 0, 0] = x
    T[0, 2, 1, 0] = x
    T[0, 1, 2, 0] = x
    T[0, 0, 1, 2] = x
    T[0, 0, 2, 1] = x
    T[1, 0, 0, 2] = x
    T[2, 0, 0, 1] = x
    T[1, 2, 2, 2] = x**2
    T[2, 1, 2, 2] = x**2
    T[2, 2, 1, 2] = x**2
    T[2, 2, 2, 1] = x**2
    T[2, 1, 1, 1] = x**2
    T[1, 2, 1, 1] = x**2
    T[1, 1, 2, 1] = x**2
    T[1, 1, 1, 2] = x**2
    '''

    
    

    

    # normalize
    val = np.einsum("ijij", T)
    T /= val
    log_factor = np.log(val)

    shape2 = (3, 3)
    s = np.zeros(shape2, dtype=float)
    s[0, 0] = s[1, 1] = s[2, 2] = 1
    

    n_spin = 1.0  # An initial tensor has one spin.
    return (T, log_factor, n_spin)

def initial_TN_O1NoShareforB(temp):
    # Z = \sum x^{length} n^{# of loops}

    shape = (2, 2, 2, 2)
    T = np.zeros(shape, dtype=float)  # [top, right, bottom, left]

    x = 1.0 / temp
    y = x**2
 

    
    T[0, 0, 0, 0] = 1
    T[1, 1, 0, 0] = T[0, 1, 1, 0] = T[0, 0, 1, 1] = T[1, 0, 0, 1] = x
    T[1, 0, 1, 0] = T[0, 1, 0, 1] = x
    T[1, 1, 1, 1] = 0

    # normalize
    val = np.einsum("ijij", T)
    T /= val
    log_factor = np.log(val)

    shape2 = (2, 2)
    s = np.zeros(shape2, dtype=float)
    s[0, 0] = s[1, 1] = 1
    

    n_spin = 1.0  # An initial tensor has one spin.
    return (T, s, log_factor, n_spin)

def initial_TN_O2NoShareforB(temp):
    # Z = \sum x^{length} n^{# of loops}

    shape = (3, 3, 3, 3)
    T = np.zeros(shape, dtype=float)  # [top, right, bottom, left]

    x = 1.0 / temp
    y = x**2
 

    
    T[0, 0, 0, 0] = 1
    T[1, 1, 0, 0] = T[0, 1, 1, 0] = T[0, 0, 1, 1] = T[1, 0, 0, 1] = x
    T[1, 0, 1, 0] = T[0, 1, 0, 1] = x
    T[1, 1, 1, 1] = 0
    T[2, 2, 0, 0] = T[0, 2, 2, 0] = T[0, 0, 2, 2] = T[2, 0, 0, 2] = x
    T[2, 0, 2, 0] = T[0, 2, 0, 2] = x

    # normalize
    val = np.einsum("ijij", T)
    T /= val
    log_factor = np.log(val)

    shape2 = (3, 3)
    s = np.zeros(shape2, dtype=float)
    s[0, 0] = s[1, 1] = s[2, 2] = 1
    

    n_spin = 1.0  # An initial tensor has one spin.
    return (T, s, log_factor, n_spin)

def initial_TN_O2CubShare(temp):
    # Z = \sum x^{length} n^{# of loops}

    shape = (3, 3, 3, 3)
    T = np.zeros(shape, dtype=float)  # [top, right, bottom, left]

    x = temp
    y = x**2
 

    
    T[0, 0, 0, 0] = 1
    T[1, 1, 0, 0] = T[0, 1, 1, 0] = T[0, 0, 1, 1] = T[1, 0, 0, 1] = x
    T[1, 0, 1, 0] = T[0, 1, 0, 1] = x
    T[2, 2, 0, 0] = T[0, 2, 2, 0] = T[0, 0, 2, 2] = T[2, 0, 0, 2] = x
    T[2, 0, 2, 0] = T[0, 2, 0, 2] = x
    T[1, 1, 1, 1] = x**2
    T[2, 2, 2, 2] = x**2

    
    
    #If you allow to share two same sites
    T[1, 1, 2, 2] = x**2
    T[2, 1, 1, 2] = x**2
    T[2, 2, 1, 1] = x**2
    T[1, 2, 2, 1] = x**2
    
    '''
    T[1, 2, 1, 2] = x**2
    T[2, 1, 2, 1] = x**2
    '''

    '''
    T[1, 2, 0, 0] = x
    T[2, 1, 0, 0] = x
    T[0, 2, 1, 0] = x
    T[0, 1, 2, 0] = x
    T[0, 0, 1, 2] = x
    T[0, 0, 2, 1] = x
    T[1, 0, 0, 2] = x
    T[2, 0, 0, 1] = x
    T[1, 2, 2, 2] = x**2
    T[2, 1, 2, 2] = x**2
    T[2, 2, 1, 2] = x**2
    T[2, 2, 2, 1] = x**2
    T[2, 1, 1, 1] = x**2
    T[1, 2, 1, 1] = x**2
    T[1, 1, 2, 1] = x**2
    T[1, 1, 1, 2] = x**2
    '''

    
    

    

    # normalize
    val = np.einsum("ijij", T)
    T /= val
    log_factor = np.log(val)

    shape2 = (3, 3)
    s = np.zeros(shape2, dtype=float)
    s[0, 0] = s[1, 1] = s[2, 2] = 1
    

    n_spin = 1.0  # An initial tensor has one spin.
    return (T, log_factor, n_spin)

def initial_TN_O1SCPforB():
    # Z = \sum x^{length} n^{# of loops}

    shape = (2, 2, 2, 2)
    T = np.zeros(shape, dtype=float)  # [top, right, bottom, left]

    '''
    x = 1.0 / temp
    y = x**2
    '''
 

    
    T[0, 0, 0, 0] = 0
    T[1, 1, 1, 1] = 2

    # normalize
    val = np.einsum("ijij", T)
    T /= val
    log_factor = np.log(val)

    shape2 = (2, 2)
    s = np.zeros(shape2, dtype=float)
    s[0, 0] = s[1, 1] = 1
    

    n_spin = 1.0  # An initial tensor has one spin.
    return (T, s, log_factor, n_spin)

def initial_TN_O2SCPforB():
    # Z = \sum x^{length} n^{# of loops}

    shape = (3, 3, 3, 3)
    T = np.zeros(shape, dtype=float)  # [top, right, bottom, left]

 

    
    T[0, 0, 0, 0] = 0
    T[1, 1, 1, 1] = T[2, 2, 2, 2] = 2
    T[2, 2, 1, 1] = T[1, 2, 2, 1] = T[1, 1, 2, 2] = T[2, 1, 1, 2] = 1
    T[1, 2, 1, 2] = T[2, 1, 2, 1] = 0

    # normalize
    val = np.einsum("ijij", T)
    T /= val
    log_factor = np.log(val)

    shape2 = (3, 3)
    s = np.zeros(shape2, dtype=float)
    s[0, 0] = s[1, 1] = s[2,2] = 1
    

    n_spin = 1.0  # An initial tensor has one spin.
    return (T, s, log_factor, n_spin)

def initial_TN_O1CPwCforB(temp):
    # Z = \sum x^{length} n^{# of loops}

    shape = (2, 2, 2, 2)
    A= np.zeros(shape, dtype=float)  # [top, right, bottom, left]

    x = temp
    y = 1/2

    A[1, 1, 0, 0] = A[1, 0, 0, 1] = np.sqrt((1-x)*y)
    A[1, 0, 1, 0] = np.sqrt(x)

    T = ncon([A, A, A, A], [[-1, 1, 2, 3], [-2, 3, 4, 5], [-3, 5, 2, 6], [-4, 6, 4, 1]])
 


    # normalize
    val = np.einsum("ijij", T)
    T /= val
    log_factor = np.log(val)

    shape2 = (2, 2)
    s = np.zeros(shape2, dtype=float)
    s[0, 0] = s[1, 1] = 1
    

    n_spin = 1.0  # An initial tensor has one spin.
    return (T, s, log_factor, n_spin)

def initial_TN_O2CPwCforB(temp1, temp2):
    # Z = \sum x^{length} n^{# of loops}

    shape = (3, 3, 3, 3)
    A = np.zeros(shape, dtype=float)  # [top, right, bottom, left]

    x = temp1
    y = (temp2 - 1/2) * (1-x) + 1/2
    #y = x**2

    A[1, 1, 0, 0] = A[1, 0, 0, 1] = np.sqrt((1-x)*y)
    A[1, 0, 1, 0] = np.sqrt(x)
    A[2, 2, 0, 0] = A[2, 0, 0, 2] = np.sqrt((1-x)*y)
    A[2, 0, 2, 0] = np.sqrt(x)

    T = ncon([A, A, A, A], [[-1, 1, 2, 3], [-2, 3, 4, 5], [-3, 5, 2, 6], [-4, 6, 4, 1]])
 

    
    '''
    T[0, 0, 0, 0] = 0
    T[1, 1, 1, 1] = T[2, 2, 2, 2] = 2 + temp
    T[2, 2, 1, 1] = T[1, 2, 2, 1] = T[1, 1, 2, 2] = T[2, 1, 1, 2] = 1
    T[1, 2, 1, 2] = T[2, 1, 2, 1] = temp
    '''

    # normalize
    val = np.einsum("ijij", T)
    T /= val
    log_factor = np.log(val)

    shape2 = (3, 3)
    s = np.zeros(shape2, dtype=float)
    s[0, 0] = s[1, 1] = s[2,2] = 1
    

    n_spin = 1.0  # An initial tensor has one spin.
    return (T, s, log_factor, n_spin)

def initial_BWTN(temp: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Initial bond-weighted tensor of the ising model on the square lattice.

    Args:
        temp: Temperature

    Returns:
        a: Initial 4-leg tensor. [top, right, bottom, left]
        w0: bond weight on the vertical bond
        w1: bond weight on the horizontal bond
        n_spin: The number of spins which contained the initial tensor.
    """
    shape = (2, 2, 2, 2)
    a = np.zeros(shape, dtype=float)  # [top, right, bottom, left]
    for idx in np.ndindex(shape):
        if sum(idx) % 2 == 0:
            a[idx] = 0.5

    c = np.cosh(1.0 / temp)
    s = np.sinh(1.0 / temp)
    w0 = 2.0 * np.diag([c, s])
    #w1 = 2.0 * np.array([c, s])

    n_spin = 1.0  # An initial tensor has one spin.
    return (a, w0, n_spin)

