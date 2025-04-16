from scipy.linalg import cho_factor, cho_solve

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