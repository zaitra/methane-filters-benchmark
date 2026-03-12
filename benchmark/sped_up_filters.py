"""
This file contains modified code from Mag1c, which is licensed under the BSD 3-Clause License. The original code can be found at https://github.com/markusfoote/mag1c
It is licensed under this license and copyright:
BSD 3-Clause License

Copyright (c) 2019, 
	Scientific Computing and Imaging Institute and 
	Utah Remote Sensing Applications Lab
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
from scipy.linalg import cho_factor, cho_solve
import numpy as np

def CEM_optimized(M, t):
    N, p = M.shape
    # Manual correlation (no mean subtraction in CEM)
    R_hat = (M.T @ M) / N  # [p, p]
    
    # Cholesky solve
    L, lower = cho_factor(R_hat, lower=True)
    Rinv_t = cho_solve((L, lower), t)  # [p]
    denom = t @ Rinv_t  # scalar
    
    return (M @ Rinv_t) / denom  # [N]

def MatchedFilterOptimized(M, t, addition=False):
    u = M.mean(axis=0)
    M_centered = M - u  # No kron needed (broadcasting)
    if addition:
        t_centered = t - u
    else:
        t_centered = t * u
    N, p = M_centered.shape
    
    # Manual covariance
    R_hat = (M_centered.T @ M_centered) / (N - 1)  # [p, p]
    
    # Cholesky solve instead of inv()
    L, lower = cho_factor(R_hat, lower=True)
    w = cho_solve((L, lower), t_centered)  # [p]
    tmp = t_centered @ w  # scalar
    
    return (M_centered @ w) / tmp  # [N]

def ACE_optimized(M, t, addition=False):
    u = M.mean(axis=0)
    M_centered = M - u  # [N, p]
    if addition:
        t_centered = t - u
    else:
        t_centered = t * u
    N, p = M_centered.shape
    
    # Efficient covariance computation
    R_hat = (M_centered.T @ M_centered) / (N - 1)
    
    # Cholesky decomposition and solves
    L, lower = cho_factor(R_hat, lower=True)
    Gt = cho_solve((L, lower), t_centered)
    GM = cho_solve((L, lower), M_centered.T)  # [p, N]
    
    # Vectorized score calculation
    tmp = t_centered @ Gt  # Scalar
    num = (t_centered @ GM) ** 2
    denom = tmp * (M_centered * GM.T).sum(axis=1)
    
    return num / denom

def mag1c_SAS(rdn_data, spec, indices):
    rdn_data_sample = rdn_data[:, indices, :]
    mu, Cit, normalizer = acrwl1mf(
        x=rdn_data_sample,
        template=spec,
        num_iter=30,
        sample=True,
    )
    mf_out = acrwl1mf_compact(
        x=rdn_data,
        normalizer=normalizer,
        num_iter=3,
        mu=mu,
        Cit=Cit,
    )
    return mf_out

def mag1c_tile(rdn_data, spec):
    mf, _ = acrwl1mf(
        x=rdn_data,
        template=spec,
        num_iter=30,
        sample=False,
    )
    return mf


def acrwl1mf(x, template, num_iter, sample=False):
    N = x.shape[1]
    
    scaling = 1e5
    epsilon = 1e-9

    # 1. Initial Statistics
    mu = np.mean(x, axis=1, keepdims=True)
    mu_T = np.swapaxes(mu, 1, 2)
    R = (x @ mu_T) / (mu @ mu_T)
    
    target = template * mu
    xmean = x - mu
    
    # C: (B, C, C) Covariance matrix
    xmean_T = np.swapaxes(xmean, 1, 2)
    C = (xmean_T @ xmean) / N

    # 2. Initial Solve (VECTORIZED OPTIMIZATION)
    target_T = np.swapaxes(target, 1, 2)
    
    # Replaced the slow 'for b in range(B)' loop with a single batched call.
    # np.linalg.solve(C, target_T) solves the system C * Cit = target_T for all batches.
    Cit = np.linalg.solve(C, target_T)

    normalizer = target @ Cit
    
    # Initial Matched Filter
    mf = ((x - mu) @ Cit) / (R * normalizer)
    mf = np.maximum(mf, 0) # ReLU

    # 3. Iterative Refinement
    for i in range(num_iter):
        regularizer = 1.0 / (R * (mf + epsilon))
        
        # Update Modified X
        modx = x - (R * mf * target)
        
        # Update Statistics
        mu = np.mean(modx, axis=1, keepdims=True)
        target = template * mu
        xmean = modx - mu
        
        xmean_T = np.swapaxes(xmean, 1, 2)
        C = (xmean_T @ xmean) / N
        
        # Update Cit (VECTORIZED OPTIMIZATION)
        target_T = np.swapaxes(target, 1, 2)
        Cit = np.linalg.solve(C, target_T)

        # Update Normalizer
        normalizer = target @ Cit
        
        # Clamp normalizer to min=1
        normalizer = np.maximum(normalizer, 1)

        # Check for sample return condition
        if sample and (i + 1 == num_iter):
            return mu, Cit, normalizer

        # Update Matched Filter with Regularization
        mf_numerator = ((x - mu) @ Cit) - regularizer
        mf = mf_numerator / (R * normalizer)
        mf = np.maximum(mf, 0) # ReLU

    mf = mf * scaling
    return mf, R

def acrwl1mf_compact(x, normalizer, num_iter, mu, Cit):
    scaling = 1e5
    epsilon = 1e-9

    # 1. Compute R
    mu_T = np.swapaxes(mu, 1, 2)
    R = (x @ mu_T) / (mu @ mu_T)

    # 2. Clamp Normalizer
    normalizer = np.maximum(normalizer, 1)

    # 3. Compute Initial Matched Filter (mf_0)
    mf_numerator = (x - mu) @ Cit
    mf_denominator = R * normalizer
    
    mf_0 = mf_numerator / mf_denominator
    mf_0 = np.maximum(mf_0, 0) # ReLU
    
    mf = mf_0.copy()

    # 4. Iterative Regularization
    for i in range(num_iter):
        regularizer = 1.0 / (R * (mf + epsilon))
        
        # mf = mf_0 - (regularizer / mf_denominator)
        mf = mf_0 - (regularizer / mf_denominator)
        
        mf = np.maximum(mf, 0) # ReLU

    # 5. Final Scaling
    mf = mf * scaling
    
    return mf


"""
Pure NumPy implementations of simple algorithms CEM, ACE and MF, is quite slower, so we did not use them after all.
def CEM_optimized_numpy(M, t):
    N, p = M.shape
    # Manual correlation (no mean subtraction in CEM)
    R_hat = (M.T @ M) / N  # [p, p]
    
    # Cholesky solve
    L = np.linalg.cholesky(R_hat)        # lower-triangular
    y = np.linalg.solve(L, t)            # solve L y = t
    Rinv_t = np.linalg.solve(L.T, y)     # solve L^T x = y
    denom = t @ Rinv_t  # scalar
    
    return (M @ Rinv_t) / denom  # [N]

def MatchedFilterOptimized_numpy(M, t, addition=False):
    u = M.mean(axis=0)
    M_centered = M - u  # No kron needed (broadcasting)
    if addition:
        t_centered = t - u
    else:
        t_centered = t * u
    N, p = M_centered.shape
    
    # Manual covariance
    R_hat = (M_centered.T @ M_centered) / (N - 1)  # [p, p]
    
    # Cholesky solve instead of inv()
    L = np.linalg.cholesky(R_hat)             # lower-triangular
    y = np.linalg.solve(L, t_centered)        # L y = t_centered
    w = np.linalg.solve(L.T, y)               # L^T w = y
    tmp = t_centered @ w  # scalar
    
    return (M_centered @ w) / tmp  # [N]

def ACE_optimized_numpy(M, t, addition=False):
    u = M.mean(axis=0)
    M_centered = M - u  # [N, p]
    if addition:
        t_centered = t - u
    else:
        t_centered = t * u
    N, p = M_centered.shape
    
    # Efficient covariance computation
    R_hat = (M_centered.T @ M_centered) / (N - 1)
    
    L = np.linalg.cholesky(R_hat)                  # lower-triangular

    # Solve R_hat x = t_centered
    y_t = np.linalg.solve(L, t_centered)
    Gt = np.linalg.solve(L.T, y_t)

    # Solve R_hat X = M_centered.T
    y_M = np.linalg.solve(L, M_centered.T)
    GM = np.linalg.solve(L.T, y_M)
    
    # Vectorized score calculation
    tmp = t_centered @ Gt  # Scalar
    num = (t_centered @ GM) ** 2
    denom = tmp * (M_centered * GM.T).sum(axis=1)
    
    return num / denom
"""