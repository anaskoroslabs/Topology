"""
kNN Opening Experiments Suite
Consolidated working experiments for TDA paper validation
Run: python knn_opening_experiments.py --experiment <name> [options]
"""

import argparse
import csv
import numpy as np
from scipy.spatial import cKDTree
from ripser import ripser
from persim import bottleneck
import time
from pathlib import Path

# =============================================================================
# MANIFOLD SAMPLERS
# =============================================================================

def sample_annulus(n, inner=0.65, outer=1.0, sigma=0.0, seed=None):
    """Sample n points uniformly from 2D annulus with optional Gaussian noise"""
    rng = np.random.default_rng(seed)
    r = np.sqrt(rng.uniform(inner**2, outer**2, n))  # uniform in area
    theta = rng.uniform(0, 2*np.pi, n)
    A = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
    if sigma > 0:
        A += rng.normal(0, sigma, A.shape)
    return A

def sample_torus_3d(n, R=1.2, r=0.45, sigma=0.0, seed=None):
    """Sample n points from 3D torus (major radius R, minor radius r)"""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2*np.pi, n)
    phi = rng.uniform(0, 2*np.pi, n)
    X = (R + r * np.cos(phi)) * np.cos(theta)
    Y = (R + r * np.cos(phi)) * np.sin(theta)
    Z = r * np.sin(phi)
    A = np.column_stack((X, Y, Z))
    if sigma > 0:
        A += rng.normal(0, sigma, A.shape)
    return A

def sample_sphere_3d(n, radius=1.0, sigma=0.0, seed=None):
    """Sample n points uniformly from 3D sphere surface"""
    rng = np.random.default_rng(seed)
    # Uniform on sphere via normal distribution
    pts = rng.normal(0, 1, size=(n, 3))
    pts = radius * pts / np.linalg.norm(pts, axis=1, keepdims=True)
    if sigma > 0:
        pts += rng.normal(0, sigma, pts.shape)
    return pts

def sample_torus_high_d(n, d_ambient=3, R=1.2, r=0.45, sigma=0.0, seed=None):
    """Sample torus and zero-pad to d_ambient dimensions"""
    A_3d = sample_torus_3d(n, R, r, sigma, seed)
    if d_ambient > 3:
        pad = np.zeros((n, d_ambient - 3))
        A = np.hstack((A_3d, pad))
    else:
        A = A_3d
    return A

# =============================================================================
# OUTLIER GENERATION
# =============================================================================

def add_isolated_outliers(clean, alpha, R=5.0, seed=None):
    """Add isolated outliers uniformly in box [-R, R]^d"""
    n_clean = len(clean)
    n_out = int(alpha * n_clean / (1 - alpha)) if alpha < 1 else n_clean
    rng = np.random.default_rng(seed)
    d = clean.shape[1]
    outliers = rng.uniform(-R, R, size=(n_out, d))
    X = np.vstack((clean, outliers))
    clean_mask = np.zeros(len(X), dtype=bool)
    clean_mask[:n_clean] = True
    return X, clean_mask, n_out

def add_clustered_outliers(clean, alpha, cluster_std=0.01, dist_factor=5.0, seed=None):
    """Add tight cluster of outliers far from manifold"""
    n_clean = len(clean)
    n_out = int(alpha * n_clean / (1 - alpha)) if alpha < 1 else n_clean
    rng = np.random.default_rng(seed)
    d = clean.shape[1]
    # Estimate manifold scale (for torus ~2*(R+r), for annulus ~2*outer)
    manifold_scale = np.max(np.linalg.norm(clean, axis=1)) * 1.2
    center = rng.uniform(-1, 1, size=d) * dist_factor * manifold_scale
    outliers = center + rng.normal(0, cluster_std, size=(n_out, d))
    X = np.vstack((clean, outliers))
    clean_mask = np.zeros(len(X), dtype=bool)
    clean_mask[:n_clean] = True
    return X, clean_mask, n_out

# =============================================================================
# KNN OPENING
# =============================================================================

def opening_knn_survivor_mask(A, r_open, k=15, tau_mult=1.0, tau_mode="median", tau_q=0.70):
    """
    kNN-depth opening: returns boolean mask of survivors
    
    Parameters:
    -----------
    A : array (n, d) - point cloud
    r_open : float - opening parameter
    k : int - number of nearest neighbors
    tau_mult : float - threshold multiplier
    tau_mode : str - 'median' or 'quantile'
    tau_q : float - quantile for quantile mode
    """
    n = len(A)
    if n <= k:
        return np.zeros(n, dtype=bool)
    
    tree = cKDTree(A)
    dists, _ = tree.query(A, k=k+1)
    knn_mean = dists[:, 1:].mean(axis=1)  # exclude self
    
    if tau_mode == "median":
        base = np.median(knn_mean)
    else:
        base = np.quantile(knn_mean, tau_q)
    
    tau0 = tau_mult * base
    allow_tau = tau0 / (1.0 + r_open / (base + 1e-12))
    return knn_mean <= allow_tau + 1e-12

# =============================================================================
# HYBRID kNN + LOF
# =============================================================================

def lof_scores(X, k_lof=20):
    """
    Local Outlier Factor scores (Breunig et al. 2000)
    Returns array of LOF scores (higher = more outlying)
    """
    n = len(X)
    if n < k_lof + 1:
        return np.ones(n)
    
    tree = cKDTree(X)
    dists, indices = tree.query(X, k=k_lof + 1)
    dists = dists[:, 1:]  # exclude self
    indices = indices[:, 1:]
    
    k_dist = dists[:, -1]  # farthest of k neighbors
    reach_dist = np.maximum(k_dist[:, None], dists)  # reachability distance
    lrd = 1.0 / (np.mean(reach_dist, axis=1) + 1e-12)  # local reachability density
    
    neighbor_lrd = lrd[indices]
    lof = np.mean(neighbor_lrd / (lrd[:, None] + 1e-12), axis=1)
    
    return lof

def hybrid_knn_lof_mask(X, r_knn=0.05, k_depth=15, tau_mult=1.3, 
                        k_lof=20, lof_threshold=1.5):
    """
    Hybrid outlier rejection: kNN-depth (coarse) + LOF (fine)
    Returns final survivor boolean mask
    """
    # Coarse kNN-depth filter
    coarse_mask = opening_knn_survivor_mask(X, r_knn, k=k_depth, tau_mult=tau_mult)
    X_coarse = X[coarse_mask]
    
    if len(X_coarse) < 10:
        return coarse_mask
    
    # Fine LOF filter on survivors
    lof = lof_scores(X_coarse, k_lof=k_lof)
    fine_mask_coarse = lof <= lof_threshold
    
    # Map back to original indices
    final_mask = np.zeros(len(X), dtype=bool)
    final_mask[coarse_mask] = fine_mask_coarse
    
    return final_mask

# =============================================================================
# PERSISTENCE COMPUTATION
# =============================================================================

def compute_h1_dgm(A, maxdim=1):
    """Compute H1 persistence diagram"""
    if len(A) < 3:
        return np.empty((0, 2))
    result = ripser(A, maxdim=maxdim, coeff=2)['dgms'][1]  # Z2 for speed
    return result

# =============================================================================
# EXPERIMENT 1: VARY SIGMA (Noise Scaling - LINEAR BOUND VALIDATION)
# =============================================================================

def experiment_vary_sigma(args):
    """
    Experiment 1: Fixed manifold, vary σ (noise), r proportional to σ
    Goal: Prove d_B ≈ C·σ (linear scaling)
    """
    print("\n" + "="*60)
    print("EXPERIMENT: Vary Sigma (Noise Scaling)")
    print("="*60)
    
    # Ground truth: high-n clean sample
    ground = sample_annulus(10000, sigma=0.0, seed=42)
    dgm_truth = compute_h1_dgm(ground)
    
    # Noise levels scaled by manifold scale
    manifold_scale = np.median(np.linalg.norm(ground[1:] - ground[:-1], axis=1))
    sigmas = np.array([0.00, 0.01, 0.02, 0.04, 0.06, 0.08]) * manifold_scale
    
    print(f"Manifold scale: {manifold_scale:.4f}")
    print(f"Sigma range: {sigmas[0]:.4f} to {sigmas[-1]:.4f}")
    
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sigma", "r_open", "trial", "survival", "dB_H1", "n_survivors"])
        
        for sigma in sigmas:
            # Set r proportional to sigma (r = 1.5*sigma, stay in-regime)
            r_open = 1.5 * sigma if sigma > 0 else 0.03
            
            for trial in range(args.trials):
                seed = 1000 + trial
                A = sample_annulus(args.n, sigma=sigma, seed=seed)
                survive = opening_knn_survivor_mask(A, r_open, k=args.k)
                A_r = A[survive]
                
                survival_rate = np.sum(survive) / len(A)
                dgm_r = compute_h1_dgm(A_r)
                dB = bottleneck(dgm_r, dgm_truth) if len(dgm_r) or len(dgm_truth) else np.inf
                
                writer.writerow([sigma, r_open, trial, survival_rate, dB, len(A_r)])
                f.flush()
        
        print(f"✓ Results saved to: {args.out_csv}")

# =============================================================================
# EXPERIMENT 2: VARY R (Opening Strength Scaling)
# =============================================================================

def experiment_vary_r(args):
    """
    Experiment 2: Fixed noise, vary r (opening parameter)
    Goal: Prove d_B ≈ C'·r + offset (linear in r)
    """
    print("\n" + "="*60)
    print("EXPERIMENT: Vary R (Opening Strength)")
    print("="*60)
    
    ground = sample_annulus(10000, sigma=0.0, seed=42)
    dgm_truth = compute_h1_dgm(ground)
    
    # Fixed moderate noise
    sigma = args.sigma
    manifold_scale = np.median(np.linalg.norm(ground[1:] - ground[:-1], axis=1))
    
    # Vary r from σ to 4σ
    r_values = np.array([1.0, 1.5, 2.0, 3.0, 4.0]) * sigma
    
    print(f"Fixed sigma: {sigma:.4f}")
    print(f"R range: {r_values[0]:.4f} to {r_values[-1]:.4f}")
    
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["r_open", "sigma", "trial", "survival", "dB_H1", "n_survivors"])
        
        for r_open in r_values:
            for trial in range(args.trials):
                seed = 2000 + trial
                A = sample_annulus(args.n, sigma=sigma, seed=seed)
                survive = opening_knn_survivor_mask(A, r_open, k=args.k)
                A_r = A[survive]
                
                survival_rate = np.sum(survive) / len(A)
                dgm_r = compute_h1_dgm(A_r)
                dB = bottleneck(dgm_r, dgm_truth) if len(dgm_r) or len(dgm_truth) else np.inf
                
                writer.writerow([r_open, sigma, trial, survival_rate, dB, len(A_r)])
                f.flush()
        
        print(f"✓ Results saved to: {args.out_csv}")

# =============================================================================
# EXPERIMENT 3: ISOLATED OUTLIERS (Breakdown Analysis)
# =============================================================================

def experiment_isolated_outliers(args):
    """
    Experiment 3: Isolated outliers (typical real-world contamination)
    Goal: Measure breakdown point α* where method fails
    """
    print("\n" + "="*60)
    print("EXPERIMENT: Isolated Outliers Breakdown")
    print("="*60)
    
    ground = sample_annulus(5000, sigma=0.0, seed=42)
    dgm_truth = compute_h1_dgm(ground)
    
    alphas = np.linspace(0.0, 0.5, 11)
    
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["alpha", "trial", "clean_survival", "outlier_survival", 
                        "overall_survival", "dB_H1", "n_survivors"])
        
        for alpha in alphas:
            print(f"  α = {alpha:.2f}", end="", flush=True)
            for trial in range(args.trials):
                seed = 3000 + trial
                clean = sample_annulus(args.n, sigma=args.sigma, seed=seed)
                X, clean_mask, n_out = add_isolated_outliers(clean, alpha, seed=seed+100)
                survive = opening_knn_survivor_mask(X, args.r, k=args.k)
                A_r = X[survive]
                
                clean_survived = np.sum(survive & clean_mask) / len(clean) if len(clean) else 0
                out_survived = np.sum(survive & ~clean_mask) / n_out if n_out else 0
                overall_survived = np.sum(survive) / len(X)
                
                dgm_r = compute_h1_dgm(A_r)
                dB = bottleneck(dgm_r, dgm_truth) if len(dgm_r) or len(dgm_truth) else np.inf
                
                writer.writerow([alpha, trial, clean_survived, out_survived, 
                               overall_survived, dB, len(A_r)])
                f.flush()
            print(" ✓")
        
        print(f"✓ Results saved to: {args.out_csv}")

# =============================================================================
# EXPERIMENT 4: CLUSTERED OUTLIERS (Adversarial)
# =============================================================================

def experiment_clustered_outliers(args):
    """
    Experiment 4: Clustered outliers (worst-case adversarial)
    Goal: Test robustness against tight outlier clusters
    """
    print("\n" + "="*60)
    print("EXPERIMENT: Clustered Outliers (Adversarial)")
    print("="*60)
    
    # Use torus for this one (2 H1 loops, more interesting)
    ground = sample_torus_3d(5000, sigma=0.0, seed=123)
    dgm_truth = compute_h1_dgm(ground)
    
    alphas = np.linspace(0.0, 0.5, 11)
    
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["alpha", "trial", "clean_survival", "outlier_survival", 
                        "overall_survival", "dB_H1", "n_survivors"])
        
        for alpha in alphas:
            print(f"  α = {alpha:.2f}", end="", flush=True)
            for trial in range(args.trials):
                seed = 4000 + trial
                clean = sample_torus_3d(args.n, sigma=args.sigma, seed=seed)
                X, clean_mask, n_out = add_clustered_outliers(clean, alpha, seed=seed+200)
                survive = opening_knn_survivor_mask(X, args.r, k=args.k)
                A_r = X[survive]
                
                clean_survived = np.sum(survive & clean_mask) / len(clean) if len(clean) else 0
                out_survived = np.sum(survive & ~clean_mask) / n_out if n_out else 0
                overall_survived = np.sum(survive) / len(X)
                
                dgm_r = compute_h1_dgm(A_r)
                dB = bottleneck(dgm_r, dgm_truth) if len(dgm_r) or len(dgm_truth) else np.inf
                
                writer.writerow([alpha, trial, clean_survived, out_survived, 
                               overall_survived, dB, len(A_r)])
                f.flush()
            print(" ✓")
        
        print(f"✓ Results saved to: {args.out_csv}")

# =============================================================================
# EXPERIMENT 5: PARAMETER SWEEP (k, tau_mode, tau_q)
# =============================================================================

def experiment_parameter_sweep(args):
    """
    Experiment 5: Mitigation strategies - vary k, tau_mode, tau_q
    Goal: Find optimal parameters for outlier rejection
    """
    print("\n" + "="*60)
    print("EXPERIMENT: Parameter Sweep (Mitigation)")
    print("="*60)
    
    ground = sample_torus_3d(5000, sigma=0.0, seed=123)
    dgm_truth = compute_h1_dgm(ground)
    
    # Parameter configurations to test
    configs = [
        {"k": 15, "tau_mode": "median", "tau_q": 0.70},
        {"k": 30, "tau_mode": "median", "tau_q": 0.70},
        {"k": 15, "tau_mode": "quantile", "tau_q": 0.80},
        {"k": 30, "tau_mode": "quantile", "tau_q": 0.80},
    ]
    
    alphas = np.linspace(0.0, 0.5, 11)
    
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["config_id", "k", "tau_mode", "tau_q", "alpha", "trial", 
                        "clean_survival", "outlier_survival", "dB_H1"])
        
        for cfg_id, cfg in enumerate(configs):
            print(f"\nConfig {cfg_id}: k={cfg['k']}, mode={cfg['tau_mode']}, q={cfg['tau_q']}")
            for alpha in alphas:
                print(f"  α = {alpha:.2f}", end="", flush=True)
                for trial in range(args.trials):
                    seed = 5000 + cfg_id * 1000 + trial
                    clean = sample_torus_3d(args.n, sigma=args.sigma, seed=seed)
                    X, clean_mask, n_out = add_clustered_outliers(clean, alpha, seed=seed+200)
                    survive = opening_knn_survivor_mask(X, args.r, k=cfg['k'], 
                                                       tau_mode=cfg['tau_mode'], 
                                                       tau_q=cfg['tau_q'])
                    A_r = X[survive]
                    
                    clean_survived = np.sum(survive & clean_mask) / len(clean)
                    out_survived = np.sum(survive & ~clean_mask) / n_out if n_out else 0
                    dgm_r = compute_h1_dgm(A_r)
                    dB = bottleneck(dgm_r, dgm_truth) if len(dgm_r) or len(dgm_truth) else np.inf
                    
                    writer.writerow([cfg_id, cfg['k'], cfg['tau_mode'], cfg['tau_q'], 
                                   alpha, trial, clean_survived, out_survived, dB])
                    f.flush()
                print(" ✓")
        
        print(f"\n✓ Results saved to: {args.out_csv}")

# =============================================================================
# EXPERIMENT 6: AMBIENT DIMENSION SWEEP
# =============================================================================

def experiment_ambient_sweep(args):
    """
    Experiment 6: Ambient dimension sweep (confirm d-independence)
    Goal: Show survival and d_B constant across ambient dimensions
    """
    print("\n" + "="*60)
    print("EXPERIMENT: Ambient Dimension Sweep")
    print("="*60)
    
    # Ground truth in 3D
    ground = sample_torus_3d(5000, sigma=0.0, seed=123)
    dgm_truth = compute_h1_dgm(ground)
    
    d_values = [3, 5, 10, 20, 50]
    alphas = [0.0, 0.2, 0.4]  # Just a few contamination levels
    
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["d_ambient", "alpha", "trial", "clean_survival", 
                        "outlier_survival", "dB_H1", "n_survivors"])
        
        for d in d_values:
            print(f"\nDimension d={d}")
            for alpha in alphas:
                print(f"  α = {alpha:.2f}", end="", flush=True)
                for trial in range(args.trials):
                    seed = 6000 + trial
                    clean = sample_torus_high_d(args.n, d_ambient=d, sigma=args.sigma, seed=seed)
                    X, clean_mask, n_out = add_clustered_outliers(clean, alpha, seed=seed+200)
                    survive = opening_knn_survivor_mask(X, args.r, k=args.k)
                    
                    # Only compute H1 on first 3 coords (intrinsic manifold)
                    A_r_3d = X[survive][:, :3]
                    
                    clean_survived = np.sum(survive & clean_mask) / len(clean)
                    out_survived = np.sum(survive & ~clean_mask) / n_out if n_out else 0
                    dgm_r = compute_h1_dgm(A_r_3d)
                    dB = bottleneck(dgm_r, dgm_truth) if len(dgm_r) or len(dgm_truth) else np.inf
                    
                    writer.writerow([d, alpha, trial, clean_survived, out_survived, 
                                   dB, len(A_r_3d)])
                    f.flush()
                print(" ✓")
        
        print(f"\n✓ Results saved to: {args.out_csv}")

# =============================================================================
# EXPERIMENT 7: HYBRID kNN+LOF
# =============================================================================

def experiment_hybrid_lof(args):
    """
    Experiment 7: Hybrid kNN+LOF vs pure kNN
    Goal: Compare rejection quality for clustered outliers
    """
    print("\n" + "="*60)
    print("EXPERIMENT: Hybrid kNN+LOF Comparison")
    print("="*60)
    
    ground = sample_torus_3d(5000, sigma=0.0, seed=123)
    dgm_truth = compute_h1_dgm(ground)
    
    alphas = np.linspace(0.0, 0.5, 11)
    
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "alpha", "trial", "clean_survival", 
                        "outlier_survival", "dB_H1", "n_survivors"])
        
        for alpha in alphas:
            print(f"  α = {alpha:.2f}", end="", flush=True)
            for trial in range(args.trials):
                seed = 7000 + trial
                clean = sample_torus_3d(args.n, sigma=args.sigma, seed=seed)
                X, clean_mask, n_out = add_clustered_outliers(clean, alpha, seed=seed+200)
                
                # Pure kNN
                survive_knn = opening_knn_survivor_mask(X, args.r, k=args.k)
                A_knn = X[survive_knn]
                clean_knn = np.sum(survive_knn & clean_mask) / len(clean)
                out_knn = np.sum(survive_knn & ~clean_mask) / n_out if n_out else 0
                dgm_knn = compute_h1_dgm(A_knn)
                dB_knn = bottleneck(dgm_knn, dgm_truth) if len(dgm_knn) or len(dgm_truth) else np.inf
                
                # Hybrid kNN+LOF
                survive_hybrid = hybrid_knn_lof_mask(X, r_knn=args.r, k_depth=args.k, 
                                                    k_lof=20, lof_threshold=1.8)
                A_hybrid = X[survive_hybrid]
                clean_hybrid = np.sum(survive_hybrid & clean_mask) / len(clean)
                out_hybrid = np.sum(survive_hybrid & ~clean_mask) / n_out if n_out else 0
                dgm_hybrid = compute_h1_dgm(A_hybrid)
                dB_hybrid = bottleneck(dgm_hybrid, dgm_truth) if len(dgm_hybrid) or len(dgm_truth) else np.inf
                
                writer.writerow(["knn", alpha, trial, clean_knn, out_knn, dB_knn, len(A_knn)])
                writer.writerow(["hybrid", alpha, trial, clean_hybrid, out_hybrid, 
                               dB_hybrid, len(A_hybrid)])
                f.flush()
            print(" ✓")
        
        print(f"\n✓ Results saved to: {args.out_csv}")

# =============================================================================
# MAIN DISPATCHER
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="kNN Opening Experiments Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available experiments:
  vary_sigma         : Vary noise σ (prove linear d_B ≈ C·σ)
  vary_r             : Vary opening r (prove linear d_B ≈ C'·r)
  isolated_outliers  : Isolated outliers breakdown test
  clustered_outliers : Clustered outliers (worst-case)
  parameter_sweep    : Vary k, tau_mode, tau_q (mitigation)
  ambient_sweep      : Ambient dimension independence test
  hybrid_lof         : Compare kNN vs kNN+LOF hybrid

Examples:
  python knn_opening_experiments.py --experiment vary_sigma
  python knn_opening_experiments.py --experiment isolated_outliers --n 1000 --trials 50
        """
    )
    
    parser.add_argument("--experiment", required=True,
                       choices=["vary_sigma", "vary_r", "isolated_outliers", 
                               "clustered_outliers", "parameter_sweep", 
                               "ambient_sweep", "hybrid_lof"],
                       help="Which experiment to run")
    
    # Common parameters
    parser.add_argument("--n", type=int, default=800, 
                       help="Number of clean points (default: 800)")
    parser.add_argument("--trials", type=int, default=30, 
                       help="Number of trials per condition (default: 30)")
    parser.add_argument("--sigma", type=float, default=0.02, 
                       help="Noise level (default: 0.02)")
    parser.add_argument("--r", type=float, default=0.08, 
                       help="Opening parameter (default: 0.08)")
    parser.add_argument("--k", type=int, default=15, 
                       help="kNN parameter (default: 15)")
    parser.add_argument("--out_csv", type=str, default=None, 
                       help="Output CSV file (default: auto-generated)")
    
    args = parser.parse_args()
    
    # Auto-generate output filename if not specified
    if args.out_csv is None:
        args.out_csv = f"{args.experiment}_results.csv"
    
    # Dispatch to appropriate experiment
    experiments = {
        "vary_sigma": experiment_vary_sigma,
        "vary_r": experiment_vary_r,
        "isolated_outliers": experiment_isolated_outliers,
        "clustered_outliers": experiment_clustered_outliers,
        "parameter_sweep": experiment_parameter_sweep,
        "ambient_sweep": experiment_ambient_sweep,
        "hybrid_lof": experiment_hybrid_lof,
    }
    
    start = time.time()
    experiments[args.experiment](args)
    elapsed = time.time() - start
    
    print(f"\n{'='*60}")
    print(f"Completed in {elapsed:.1f}s")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
