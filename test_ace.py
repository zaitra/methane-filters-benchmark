import spectral as spy
import numpy as np
import time
import argparse
from scipy.linalg import cho_factor, cho_solve, inv
import scipy.linalg as lin
import torch
import torch.nn as nn

class ACEModel(nn.Module):
    def __init__(self):
        super(ACEModel, self).__init__()

    def forward(self, M, t):
        """
        Optimized Adaptive Coherent Estimator (ACE) using Cholesky decomposition in PyTorch.
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
        
        return num / denom

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
    N, p = M.shape
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
    parser = argparse.ArgumentParser(description="Compute ACE for a given hyperspectral image.")
    parser.add_argument("hdr_path", type=str, help="Path to the hyperspectral HDR file.")
    parser.add_argument("methane_spectrum", type=str, help="Path to the methane spectrum numpy file (.npy).")
    
    args = parser.parse_args()
    
    # Load hyperspectral image
    print(f"Loading hyperspectral image from {args.hdr_path}...")
    hyperspectral_img_filtered = load_hyperspectral_image(args.hdr_path)
    
    # Load methane spectrum
    print(f"Loading methane spectrum from {args.methane_spectrum}...")
    methane_spectrum_filtered = np.load(args.methane_spectrum)

    # Start timing
    hyperspectral_img_filtered = hyperspectral_img_filtered.squeeze()
    measure_process("ACE_original", ACE_original, hyperspectral_img_filtered, methane_spectrum_filtered)
    measure_process("ACE_without_cholesky", ACE_without_cholesky, hyperspectral_img_filtered, methane_spectrum_filtered)
    measure_process("ACE_sped_up", ACE_sped_up, hyperspectral_img_filtered, methane_spectrum_filtered)
    model = ACEModel()
    measure_process("Torch model", model, hyperspectral_img_filtered, methane_spectrum_filtered)
    torch.set_num_threads(4)  # Use 4 threads for CPU
    torch.backends.mkldnn.enabled = True  # Enable oneDNN optimizations
    measure_process("Torch model", model, hyperspectral_img_filtered, methane_spectrum_filtered)


if __name__ == "__main__":
    main()
