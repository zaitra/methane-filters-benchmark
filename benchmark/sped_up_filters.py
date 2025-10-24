from scipy.linalg import cho_factor, cho_solve
import numpy as np
from typing import Tuple

def CEM_optimized(M, t):
    N, p = M.shape
    # Manual correlation (no mean subtraction in CEM)
    R_hat = (M.T @ M) / N  # [p, p]
    
    # Cholesky solve
    L, lower = cho_factor(R_hat, lower=True)
    Rinv_t = cho_solve((L, lower), t)  # [p]
    denom = t @ Rinv_t  # scalar
    
    return (M @ Rinv_t) / denom  # [N]

def MatchedFilterOptimized(M, t):
    u = M.mean(axis=0)
    M_centered = M - u  # No kron needed (broadcasting)
    t_centered = t - u
    N, p = M_centered.shape
    
    # Manual covariance
    R_hat = (M_centered.T @ M_centered) / (N - 1)  # [p, p]
    
    # Cholesky solve instead of inv()
    L, lower = cho_factor(R_hat, lower=True)
    w = cho_solve((L, lower), t_centered)  # [p]
    tmp = t_centered @ w  # scalar
    
    return (M_centered @ w) / tmp  # [N]

def ACE_optimized(M, t):
    u = M.mean(axis=0)
    M_centered = M - u  # [N, p]
    t_centered = t - u  # [p]
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

def compute_mag1c_sas(rdn_data, spec, indices):
    rdn_data_sample = rdn_data[:, indices, :]
    mu, Cit, normalizer = acrwl1mf_numpy_cleaned(
        x=rdn_data_sample,
        template=spec,
        num_iter=30,
        sample=True,
    )
    
    mf_out = acrwl1mf_compact_numpy_cleaned(
        x=rdn_data,
        normalizer=normalizer,
        num_iter=3,
        mu=mu,
        Cit=Cit,
    )
    return mf_out

def acrwl1mf_numpy_cleaned(
    x: np.ndarray,
    template: np.ndarray,
    num_iter: int,
    sample: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate the albedo-corrected reweighted-L1 matched filter on radiance data using NumPy."""
    dtype = x.dtype
    N = x.shape[1]  # number of samples
    regularizer = np.zeros((x.shape[0], x.shape[1], 1), dtype=dtype)
    modx = x
    
    scaling = np.array(1e5, dtype=dtype)
    epsilon = np.array(1e-9, dtype=dtype)
    
    # Initialize with normal robust matched filter
    mu = np.mean(modx, axis=1, keepdims=True)  # [batch x 1 x spectrum]
    
    # [b x p x s] * [b x s x 1] = [b x p x 1]
    numerator = np.matmul(x, np.transpose(mu, (0, 2, 1)))
    # [b x 1 x s] * [b x s x 1] = [b x 1 x 1]
    denominator = np.matmul(mu, np.transpose(mu, (0, 2, 1)))
    R = np.divide(numerator, denominator)
    
    target = template * mu  # [1 x 1 x s] * [b x 1 x s] = [b x 1 x s]
    xmean = modx - mu
    
    # [b x s x p] * [b x p x s] = [b x s x s]
    C = np.matmul(np.transpose(xmean, (0, 2, 1)), xmean) / N

    
    # Solve using Cholesky decomposition
    Cit = np.zeros_like(np.transpose(target, (0, 2, 1)))
    for i in range(C.shape[0]):
        L = np.linalg.cholesky(C[i])
        # Solve L L^T Cit = target^T
        y = np.linalg.solve(L, np.transpose(target, (0, 2, 1))[i])
        Cit[i] = np.linalg.solve(L.T, y)
    
    normalizer = np.matmul(target, Cit)  # [b x 1 x s] * [b x s x 1] = [b x 1 x 1]
    
    # [b x p x s] * [b x s x 1] = [b x p x 1]
    numerator_mf = np.matmul(x - mu, Cit)
    denominator_mf = R * normalizer
    mf = np.divide(numerator_mf, denominator_mf)
    
    mf = np.maximum(mf, 0)  # ReLU equivalent
    
    # Reweighted L1 Algorithm
    for i in range(num_iter):
        # Calculate new regularizer weights
        regularizer = np.reciprocal(R * mf + epsilon)
        
        # Re-calculate statistics
        modx = x - (R * mf * target)
        mu = np.mean(modx, axis=1, keepdims=True)
        target = template * mu
        xmean = modx - mu
        
        C = np.matmul(np.transpose(xmean, (0, 2, 1)), xmean) / N
    
        
        # Update Cit using Cholesky
        for j in range(C.shape[0]):
            L = np.linalg.cholesky(C[j])
            y = np.linalg.solve(L, np.transpose(target, (0, 2, 1))[j])
            Cit[j] = np.linalg.solve(L.T, y)
        
        # Compute matched filter with regularization
        normalizer = np.matmul(target, Cit)
        if np.any(normalizer < 1):
            normalizer = np.clip(normalizer, 1, None)
            
        if num_iter == i + 1 and sample:
            return mu, Cit, normalizer
            
        numerator_mf = np.matmul(x - mu, Cit) - regularizer
        denominator_mf = R * normalizer
        mf = np.divide(numerator_mf, denominator_mf)
        
        mf = np.maximum(mf, 0)
    
    mf = mf * scaling
    return mf, R, normalizer

def acrwl1mf_compact_numpy_cleaned(
    x: np.ndarray,
    normalizer: np.ndarray,
    num_iter: int,
    mu: np.ndarray,
    Cit: np.ndarray,
    ):
    """Calculate the albedo-corrected reweighted-L1 matched filter on radiance data using NumPy."""
    
    dtype = x.dtype
    
    regularizer = np.zeros((x.shape[0], x.shape[1], 1), dtype=dtype)
    scaling = np.array(1e5, dtype=dtype)
    epsilon = np.array(1e-9, dtype=dtype)
    
    # Initialize with normal robust matched filter
    numerator = np.matmul(x, np.transpose(mu, (0, 2, 1)))
    denominator = np.matmul(mu, np.transpose(mu, (0, 2, 1)))
    R = np.divide(numerator, denominator)
    
    if np.any(normalizer < 1):
        normalizer = np.clip(normalizer, 1, None)
    
    numerator_mf = np.matmul(x - mu, Cit)
    denominator_mf = R * normalizer
    mf_0 = np.divide(numerator_mf, denominator_mf)
    
    mf_0 = np.maximum(mf_0, 0)
    
    mf = mf_0.copy()
    
    # Reweighted L1 Algorithm
    for i in range(num_iter):
        # Calculate new regularizer weights
        regularizer = np.reciprocal(R * mf + epsilon)
        
        mf = mf_0 - np.divide(regularizer, R * normalizer)
        
        mf = np.maximum(mf, 0)
    
    mf = mf * scaling
    return mf