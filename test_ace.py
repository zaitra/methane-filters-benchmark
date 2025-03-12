import spectral as spy
import numpy as np
import time
import argparse
from scipy.linalg import cho_factor, cho_solve, inv
import scipy.linalg as lin
import torch
import torch.nn as nn
from numba import njit
import os


class ACEModel(nn.Module):
    def __init__(self):
        super(ACEModel, self).__init__()

    def forward(self, M, t):
        u = M.mean(dim=0)
        M_centered = M - u  # [N, p]
        t_centered = t - u  # [p]
        N, p = M_centered.shape
        
        # Manual covariance matrix
        R_hat = (M_centered.T @ M_centered) / (N - 1)
        
        # Cholesky decomposition and solves
        L = torch.linalg.cholesky(R_hat)
        Gt = torch.cholesky_solve(t_centered.unsqueeze(1), L).squeeze(1)
        GM = torch.cholesky_solve(M_centered.T, L)  # [p, N]
        
        # Compute scores
        tmp = torch.dot(t_centered, Gt)
        num = torch.matmul(t_centered, GM).pow(2)
        denom = tmp * (M_centered * GM.T).sum(dim=1)
        
        return num / denom
        """
        Optimized Adaptive Coherent Estimator (ACE) using Cholesky decomposition in PyTorch.
        """
        """
        # Remove mean
        u = torch.mean(M, dim=0)
        M_centered = M - u
        t_centered = t - u

        # Estimate covariance matrix
        R_hat = torch.cov(M_centered.T, correction=1)
        
        # Cholesky decomposition
        L = torch.linalg.cholesky(R_hat)
        
        # Solve for Gt and GM
        Gt = torch.cholesky_solve(t_centered.unsqueeze(1), L).squeeze(1)  # [p]
        GM = torch.cholesky_solve(M_centered.T, L)  # [p, N]
        
        # Compute ACE scores
        tmp = torch.dot(t_centered, Gt)  # Scalar
        num = torch.matmul(t_centered, GM).pow(2)  # [N]
        denom = tmp * (M_centered * GM.T).sum(dim=1)  # [N]
        
        return num / denom"""

def ACE_without_cholesky(M, t):
    """
    Optimized Adaptive Coherent Estimator (ACE) for target detection.
    """
    N, p = M.shape
    # Remove mean
    u = M.mean(axis=0)
    M = M - u
    t = t - u

    # Estimate covariance and compute its inverse
    R_hat = np.cov(M, rowvar=False)
    G = inv(R_hat)

    # Precompute terms
    tmp = t.T @ G @ t  # Scalar
    dot_G_M = G @ M.T   # (p, N)
    
    num = (t @ dot_G_M) ** 2  # Vectorized numerator
    denom = tmp * np.einsum('ij,ji->i', M, dot_G_M)  # Vectorized denominator

    return num / denom

def ACE_original(M, t):
    """
    Performs the adaptive cosin/coherent estimator algorithm for target
    detection.

    Parameters:
        M: `numpy array`
            2d matrix of HSI data (N x p).

        t: `numpy array`
            A target endmember (p).

    Returns: `numpy array`
        Vector of detector output (N).

    References:
      X Jin, S Paswater, H Cline.  "A Comparative Study of Target Detection
      Algorithms for Hyperspectral Imagery."  SPIE Algorithms and Technologies
      for Multispectral, Hyperspectral, and Ultraspectral Imagery XV.  Vol
      7334.  2009.
    """
    N, p = M.shape
    # Remove mean from data
    u = M.mean(axis=0)
    M = M - np.kron(np.ones((N, 1)), u)
    t = t - u;

    R_hat = np.cov(M.T)
    G = lin.inv(R_hat)

    results = np.zeros(N, dtype=np.float32)
    ##% From Broadwater's paper
    ##%tmp = G*S*inv(S.'*G*S)*S.'*G;
    tmp = np.array(np.dot(t.T, np.dot(G, t)))
    dot_G_M = np.dot(G, M[0:,:].T)
    num = np.square(np.dot(t, dot_G_M))
    for k in range(N):
        denom = np.dot(tmp, np.dot(M[k], dot_G_M[:,k]))
        results[k] = num[k] / denom
    return results


def ACE_sped_up(M, t):
    """
    Optimized Adaptive Coherent Estimator (ACE) using Cholesky decomposition.
    """
    # Remove mean
    u = M.mean(axis=0)
    M = M - u
    t = t - u

    # Estimate covariance and compute its inverse using Cholesky decomposition
    R_hat = np.cov(M, rowvar=False)
    L, lower = cho_factor(R_hat, lower=True)  # Cholesky decomposition
    Gt = cho_solve((L, lower), t)  # Solve G @ x = t
    GM = cho_solve((L, lower), M.T)  # Solve G @ X = M.T

    # Compute ACE score
    tmp = t @ Gt  # Scalar
    num = (t @ GM) ** 2  # Vectorized numerator
    denom = tmp * np.einsum('ij,ji->i', M, GM)  # Vectorized denominator

    return num / denom

def ACE_sped_up_2(M, t):
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

def CEM_original(M, t):
    """
    Performs the constrained energy minimization algorithm for target
    detection.

    Parameters:
        M: `numpy array`
            2d matrix of HSI data (N x p).

        t: `numpy array`
            A target endmember (p).

    Returns: `numpy array`
        Vector of detector output (N).

    References:
        Qian Du, Hsuan Ren, and Chein-I Cheng. A Comparative Study of
        Orthogonal Subspace Projection and Constrained Energy Minimization.
        IEEE TGRS. Volume 41. Number 6. June 2003.
    """
    def corr(M):
        p, N = M.shape
        return np.dot(M, M.T) / N

    N, p = M.shape
    R_hat = corr(M.T)
    Rinv = lin.inv(R_hat)
    denom = np.dot(t.T, np.dot(Rinv, t))
    t_Rinv = np.dot(t.T, Rinv)
    return np.dot(t_Rinv , M[0:,:].T) / denom

def CEM_optimized(M, t):
    N, p = M.shape
    # Manual correlation (no mean subtraction in CEM)
    R_hat = (M.T @ M) / N  # [p, p]
    
    # Cholesky solve
    L, lower = cho_factor(R_hat, lower=True)
    Rinv_t = cho_solve((L, lower), t)  # [p]
    denom = t @ Rinv_t  # scalar
    
    return (M @ Rinv_t) / denom  # [N]

def MatchedFilter_original(M, t):
    """
    Performs the matched filter algorithm for target
    detection.

    Parameters:
        M: `numpy array`
            2d matrix of HSI data (N x p).

        t: `numpy array`
            A target endmember (p).

    Returns: `numpy array`
        Vector of detector output (N).

    References:
      X Jin, S Paswater, H Cline.  "A Comparative Study of Target Detection
     Algorithms for Hyperspectral Imagery."  SPIE Algorithms and Technologies
     for Multispectral, Hyperspectral, and Ultraspectral Imagery XV.  Vol
     7334.  2009.
    """

    N, p = M.shape
    # Remove mean from data
    u = M.mean(axis=0)
    M = M - np.kron(np.ones((N, 1)), u)
    t = t - u;

    R_hat = np.cov(M.T)
    G = lin.inv(R_hat)

    tmp = np.array(np.dot(t.T, np.dot(G, t)))
    w = np.dot(G, t)
    return np.dot(w, M.T) / tmp

def MatchedFilter_optimized(M, t):
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

def load_hyperspectral_image(hdr_path):
    """Load hyperspectral image using Spectral Python (SPy)."""
    img = spy.open_image(hdr_path).load()
    return img

def compute_ace(hyperspectral_img, methane_spectrum):
    """Compute Adaptive Cosine Estimator (ACE) for hyperspectral image."""
    ace_result = spy.ace(hyperspectral_img, methane_spectrum)
    return ace_result

def measure_process(name, function, hyperspectral_img_filtered, methane_spectrum_filtered):
    print(f"Computing {name}...")
    start_time = time.time()

    # Compute ACE
    result = function(hyperspectral_img_filtered, methane_spectrum_filtered)

    # End timing
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"{name} Computation Done! Processing time: {elapsed_time:.4f} seconds")


def main():
    os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
    os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4 
    os.environ["MKL_NUM_THREADS"] = "4" # export MKL_NUM_THREADS=6
    os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
    os.environ["NUMEXPR_NUM_THREADS"] = "4" # export NUMEXPR_NUM_THREADS=6
    parser = argparse.ArgumentParser(description="Compute ACE for a given hyperspectral image.")
    
    # Make hdr_path and methane_spectrum optional
    parser.add_argument("hdr_path", type=str, nargs="?", default=None, help="Path to the hyperspectral HDR file.")
    parser.add_argument("methane_spectrum", type=str, nargs="?", default=None, help="Path to the methane spectrum numpy file (.npy).")
    
    # Shape argument
    parser.add_argument("--random", type=int, nargs=3, default=(512, 512, 50),
                        help="If not specified the path to real image, random image with specified shape will be generated, type it as H W C (default: 512 512 50)")
    
    args = parser.parse_args()

    if args.hdr_path and args.methane_spectrum:
        # Load hyperspectral image
        print(f"Loading hyperspectral image from {args.hdr_path}...")
        hyperspectral_img_filtered = load_hyperspectral_image(args.hdr_path)
        
        # Load methane spectrum
        print(f"Loading methane spectrum from {args.methane_spectrum}...")
        methane_spectrum_filtered = np.load(args.methane_spectrum).astype(np.float32)
        hyperspectral_img_filtered = hyperspectral_img_filtered.squeeze().astype(np.float32)
    
    else:
        # Use random data generation if no file paths are provided
        H, W, C = args.random  # Unpack the shape from arguments
        print(f"Generating random hyperspectral image with shape: {H}, {W}, {C}...")
        hyperspectral_img_filtered = np.random.rand(H, W, C).astype(np.float32).reshape(-1,C)  # Generate random hyperspectral image
        print(f"Generated hyperspectral image with shape: {hyperspectral_img_filtered.shape}")
        
        methane_spectrum_filtered = np.random.rand(C).astype(np.float32)  # Random methane spectrum
        print(f"Generated random methane spectrum with shape: {methane_spectrum_filtered.shape}")
    
    hyperspectral_img_filtered = np.ascontiguousarray(hyperspectral_img_filtered, dtype=np.float32)
    methane_spectrum_filtered = np.ascontiguousarray(methane_spectrum_filtered, dtype=np.float32)
    print(hyperspectral_img_filtered.shape)
    print(methane_spectrum_filtered.shape)
    measure_process("ACE_original", ACE_original, hyperspectral_img_filtered, methane_spectrum_filtered)
    #measure_process("ACE_without_cholesky", ACE_without_cholesky, hyperspectral_img_filtered, methane_spectrum_filtered)
    #measure_process("ACE_sped_up", ACE_sped_up, hyperspectral_img_filtered, methane_spectrum_filtered)
    measure_process("ACE_sped_up_2", ACE_sped_up_2, hyperspectral_img_filtered, methane_spectrum_filtered)
    measure_process("MatchedFilter_original", MatchedFilter_original, hyperspectral_img_filtered, methane_spectrum_filtered)
    measure_process("MatchedFilter_optimized", MatchedFilter_optimized, hyperspectral_img_filtered, methane_spectrum_filtered)
    measure_process("CEM_original", CEM_original, hyperspectral_img_filtered, methane_spectrum_filtered)
    measure_process("CEM_optimized", CEM_optimized, hyperspectral_img_filtered, methane_spectrum_filtered)
    hyperspectral_img_filtered = torch.from_numpy(hyperspectral_img_filtered.copy()).contiguous().to(torch.float32)
    methane_spectrum_filtered = torch.from_numpy(methane_spectrum_filtered.copy()).contiguous().to(torch.float32)
    model = ACEModel()
    measure_process("Torch model", model, hyperspectral_img_filtered, methane_spectrum_filtered)
    torch.set_num_threads(4)  # Use 4 threads for CPU
    torch.backends.mkldnn.enabled = True  # Enable oneDNN optimizations
    measure_process("Torch model with mkldnn enabled (4 threads)", model, hyperspectral_img_filtered, methane_spectrum_filtered)
    with torch.jit.optimized_execution(True):
        scripted_model = torch.jit.trace(model, example_inputs=(hyperspectral_img_filtered, methane_spectrum_filtered))
    # Save the scripted model
    scripted_model.save("ace_model.pt")
    loaded_model = torch.jit.load("ace_model.pt")
    measure_process("Torch model with JIT", loaded_model, hyperspectral_img_filtered, methane_spectrum_filtered)


if __name__ == "__main__":
    main()
