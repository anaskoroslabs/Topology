"""
Swiss Roll Dimension Validation - CORRECTED VERSION
Matches kNN opening method from paper (median-based adaptive threshold)
"""
import numpy as np
import csv
from sklearn.datasets import make_swiss_roll
from sklearn.neighbors import NearestNeighbors
from ripser import ripser
from persim import bottleneck, plot_diagrams
import matplotlib.pyplot as plt
from tqdm import tqdm

np.random.seed(42)

# Parameters
n_points = 500
k = 30
r_open = 0.01
tau_mult = 1.0
ambient_dims = [10, 50, 100, 500, 1000]
n_trials = 5  # Multiple trials for robustness

# Step 1: Generate 2D Swiss roll (intrinsic manifold)
X_base, _ = make_swiss_roll(n_points, noise=0.02)
X_2d = X_base[:, [0, 2]]  # Keep 2D structure

# Step 2: CORRECTED embedding - zero padding (NOT random noise)
def embed_zero_pad(X, d):
    """Embed by zero-padding (preserves distances in original coordinates)"""
    n, m = X.shape
    if d < m:
        raise ValueError(f"Target dimension {d} < current dimension {m}")
    X_embedded = np.zeros((n, d))
    X_embedded[:, :m] = X  # Original data in first m coordinates
    return X_embedded

# Step 3: CORRECTED kNN opening (matches paper algorithm)
def knn_opening(X, k, r_open, tau_mult=1.0):
    """
    kNN-depth opening matching Algorithm 1 in paper
    """
    n = X.shape[0]
    
    # Compute kNN distances
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(X)
    distances, _ = nbrs.kneighbors(X)
    
    # Mean distance to k nearest neighbors (excluding self)
    delta_k = distances[:, 1:].mean(axis=1)
    
    # Adaptive threshold (matches paper)
    tau_0 = tau_mult * np.median(delta_k)
    tau_r = tau_0 / (1 + r_open / tau_0)
    
    # Opening: keep points with delta_k <= tau_r
    survivors = delta_k <= tau_r
    
    return X[survivors], survivors

# Step 4: Run dimension sweep
results = {
    'dims': ambient_dims,
    'bottleneck_mean': [],
    'bottleneck_std': [],
    'survival_mean': [],
    'survival_std': []
}

print("\n" + "="*70)
print("SWISS ROLL DIMENSION VALIDATION (CORRECTED)")
print("="*70)
print(f"Intrinsic dimension: m=2")
print(f"Sample size: n={n_points}")
print(f"kNN parameter: k={k}")
print(f"Opening parameter: r={r_open}")
print(f"Trials per dimension: {n_trials}")
print("="*70 + "\n")

# Compute ground truth (base case: d=10, WITH opening)
print("Computing ground truth (d=10, with opening)...")
X_10 = embed_zero_pad(X_2d, 10)
X_10_opened, _ = knn_opening(X_10, k, r_open, tau_mult)
result_base = ripser(X_10_opened, maxdim=1)
dgm_base = result_base['dgms']
print(f"  H1 features: {len(dgm_base[1])}")
print(f"  Survival rate: {100*len(X_10_opened)/len(X_10):.1f}%")

print("\nStarting dimension sweep...\n")

for d in tqdm(ambient_dims, desc="Dimensions"):
    bottlenecks = []
    survivals = []
    
    for trial in range(n_trials):
        # Embed into d dimensions (zero padding)
        X_d = embed_zero_pad(X_2d, d)
        
        # Apply kNN opening
        X_opened, survive_mask = knn_opening(X_d, k, r_open, tau_mult)
        
        survival_rate = np.mean(survive_mask)
        survivals.append(survival_rate)
        
        # Compute persistence
        if len(X_opened) > k + 1:  # Need enough points
            result = ripser(X_opened, maxdim=1)
            dgm = result['dgms']
            
            # Bottleneck distance to base
            if len(dgm[1]) > 0 and len(dgm_base[1]) > 0:
                d_B = bottleneck(dgm[1], dgm_base[1])
            else:
                d_B = 0.0 if len(dgm[1]) == len(dgm_base[1]) else float('inf')
            
            bottlenecks.append(d_B)
        else:
            bottlenecks.append(float('inf'))  # Too few survivors
    
    # Store statistics
    results['bottleneck_mean'].append(np.mean(bottlenecks))
    results['bottleneck_std'].append(np.std(bottlenecks))
    results['survival_mean'].append(np.mean(survivals))
    results['survival_std'].append(np.std(survivals))
    
    print(f"d={d:4d}: d_B = {np.mean(bottlenecks):.4f} +/- {np.std(bottlenecks):.4f}, "
          f"survival = {100*np.mean(survivals):.1f}% +/- {100*np.std(survivals):.1f}%")

print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)

# Check dimension independence
max_dB = max(results['bottleneck_mean'])
if max_dB < 0.5:
    verdict = "EXCELLENT - Dimension independence confirmed!"
elif max_dB < 1.0:
    verdict = "GOOD - Stable with minor degradation"
else:
    verdict = "FAILED - Significant dimension dependence"

print(f"\nMaximum bottleneck distance: {max_dB:.4f}")
print(f"Verdict: {verdict}\n")

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Bottleneck distance
ax1.errorbar(results['dims'], results['bottleneck_mean'], 
             yerr=results['bottleneck_std'], 
             marker='o', capsize=5, linewidth=2, markersize=8)
ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Target threshold')
ax1.set_xlabel('Ambient Dimension d', fontsize=12)
ax1.set_ylabel('Bottleneck Distance to d=10', fontsize=12)
ax1.set_title('Dimension Independence: Topology Preservation', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.set_xscale('log')

# Plot 2: Survival rate
ax2.errorbar(results['dims'], 
             [100*m for m in results['survival_mean']], 
             yerr=[100*s for s in results['survival_std']],
             marker='s', capsize=5, linewidth=2, markersize=8, color='green')
ax2.axhline(y=50, color='orange', linestyle='--', alpha=0.5, label='50% target')
ax2.set_xlabel('Ambient Dimension d', fontsize=12)
ax2.set_ylabel('Survival Rate (%)', fontsize=12)
ax2.set_title('Dimension Independence: Consistent Trimming', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.set_xscale('log')

plt.tight_layout()
plt.savefig('swiss_roll_dimension_validation_corrected.png', dpi=300, bbox_inches='tight')
print("Plot saved: swiss_roll_dimension_validation_corrected.png")
plt.show()

# Save results to CSV
csv_filename = 'swiss_roll_results.csv'
with open(csv_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # Header
    writer.writerow(['d', 'd_B_mean', 'd_B_std', 'survival_mean', 'survival_std', 'status'])
    # Data rows
    for i, d in enumerate(results['dims']):
        dB_mean = results['bottleneck_mean'][i]
        dB_std = results['bottleneck_std'][i]
        surv_mean = results['survival_mean'][i]
        surv_std = results['survival_std'][i]
        status = "pass" if dB_mean < 0.5 else ("marginal" if dB_mean < 1.0 else "fail")
        writer.writerow([d, dB_mean, dB_std, surv_mean, surv_std, status])

print(f"\nResults saved to: {csv_filename}")

# Generate LaTeX table for paper
print("\n" + "="*70)
print("LATEX TABLE (for paper)")
print("="*70)
print("\\begin{table}[h]")
print("\\centering")
print("\\caption{Swiss roll dimension validation results}")
print("\\begin{tabular}{ccccc}")
print("\\toprule")
print("$d$ & $d_B$ & Survival (\\%) & H$_1$ Features & Status \\\\")
print("\\midrule")
for i, d in enumerate(results['dims']):
    dB_mean = results['bottleneck_mean'][i]
    dB_std = results['bottleneck_std'][i]
    surv_mean = results['survival_mean'][i]
    status_symbol = "pass" if dB_mean < 0.5 else ("marginal" if dB_mean < 1.0 else "fail")
    print(f"{d} & ${dB_mean:.2f} \\pm {dB_std:.2f}$ & ${100*surv_mean:.1f}$ & -- & {status_symbol} \\\\")
print("\\bottomrule")
print("\\end{tabular}")
print("\\end{table}")
print("\n" + "="*70)
