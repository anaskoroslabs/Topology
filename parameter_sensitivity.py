"""
Parameter Sensitivity Analysis
Tests stability of kNN opening over ranges of k and r
"""
import numpy as np
from sklearn.datasets import make_swiss_roll
from sklearn.neighbors import NearestNeighbors
from ripser import ripser
from persim import bottleneck
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm

np.random.seed(42)

def embed_zero_pad(X, d):
    n, m = X.shape
    X_embedded = np.zeros((n, d))
    X_embedded[:, :m] = X
    return X_embedded

def knn_opening(X, k, r_open, tau_mult=1.0):
    n = X.shape[0]
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(X)
    distances, _ = nbrs.kneighbors(X)
    delta_k = distances[:, 1:].mean(axis=1)
    tau_0 = tau_mult * np.median(delta_k)
    tau_r = tau_0 / (1 + r_open / tau_0)
    survivors = delta_k <= tau_r
    return X[survivors], survivors

# Parameters
n_points = 500
test_dimension = 50  # High dimension to stress-test
k_values = [10, 15, 20, 30, 40, 50]
r_values = [0.005, 0.01, 0.02, 0.05, 0.1]
n_trials = 3

# Generate Swiss roll
X_base, _ = make_swiss_roll(n_points, noise=0.02)
X_2d = X_base[:, [0, 2]]

# Baseline (k=30, r=0.01)
k_baseline = 30
r_baseline = 0.01

X_baseline = embed_zero_pad(X_2d, test_dimension)
X_baseline_opened, _ = knn_opening(X_baseline, k_baseline, r_baseline)
result_baseline = ripser(X_baseline_opened, maxdim=1)
dgm_baseline = result_baseline['dgms']

print("\n" + "="*70)
print("PARAMETER SENSITIVITY ANALYSIS")
print("="*70)
print(f"Test dimension: d={test_dimension}")
print(f"Baseline: k={k_baseline}, r={r_baseline}")
print(f"Baseline H1 features: {len(dgm_baseline[1])}")
print(f"Baseline survival: {100*len(X_baseline_opened)/len(X_baseline):.1f}%")
print("="*70 + "\n")

# Test k sensitivity (fixed r)
print("Testing k sensitivity (r=0.01):")
k_results = {
    'k_values': k_values,
    'bottleneck_mean': [],
    'bottleneck_std': [],
    'survival_mean': [],
    'survival_std': []
}

for k in tqdm(k_values, desc="k values"):
    bottlenecks = []
    survivals = []
    
    for trial in range(n_trials):
        np.random.seed(42 + trial + k)
        X_test = embed_zero_pad(X_2d, test_dimension)
        X_opened, survive_mask = knn_opening(X_test, k, r_baseline)
        
        survivals.append(np.mean(survive_mask))
        
        if len(X_opened) > k + 1:
            result = ripser(X_opened, maxdim=1)
            dgm = result['dgms']
            
            if len(dgm[1]) > 0 and len(dgm_baseline[1]) > 0:
                d_B = bottleneck(dgm[1], dgm_baseline[1])
            else:
                d_B = 0.0 if len(dgm[1]) == len(dgm_baseline[1]) else 1.0
            
            bottlenecks.append(d_B)
        else:
            bottlenecks.append(1.0)
    
    k_results['bottleneck_mean'].append(np.mean(bottlenecks))
    k_results['bottleneck_std'].append(np.std(bottlenecks))
    k_results['survival_mean'].append(np.mean(survivals))
    k_results['survival_std'].append(np.std(survivals))
    
    print(f"  k={k:2d}: d_B={np.mean(bottlenecks):.3f}±{np.std(bottlenecks):.3f}, "
          f"survival={100*np.mean(survivals):.1f}%±{100*np.std(survivals):.1f}%")

# Test r sensitivity (fixed k)
print("\nTesting r sensitivity (k=30):")
r_results = {
    'r_values': r_values,
    'bottleneck_mean': [],
    'bottleneck_std': [],
    'survival_mean': [],
    'survival_std': []
}

for r in tqdm(r_values, desc="r values"):
    bottlenecks = []
    survivals = []
    
    for trial in range(n_trials):
        np.random.seed(42 + trial + int(r*1000))
        X_test = embed_zero_pad(X_2d, test_dimension)
        X_opened, survive_mask = knn_opening(X_test, k_baseline, r)
        
        survivals.append(np.mean(survive_mask))
        
        if len(X_opened) > k_baseline + 1:
            result = ripser(X_opened, maxdim=1)
            dgm = result['dgms']
            
            if len(dgm[1]) > 0 and len(dgm_baseline[1]) > 0:
                d_B = bottleneck(dgm[1], dgm_baseline[1])
            else:
                d_B = 0.0 if len(dgm[1]) == len(dgm_baseline[1]) else 1.0
            
            bottlenecks.append(d_B)
        else:
            bottlenecks.append(1.0)
    
    r_results['bottleneck_mean'].append(np.mean(bottlenecks))
    r_results['bottleneck_std'].append(np.std(bottlenecks))
    r_results['survival_mean'].append(np.mean(survivals))
    r_results['survival_std'].append(np.std(survivals))
    
    print(f"  r={r:.3f}: d_B={np.mean(bottlenecks):.3f}±{np.std(bottlenecks):.3f}, "
          f"survival={100*np.mean(survivals):.1f}%±{100*np.std(survivals):.1f}%")

print("\n" + "="*70)
print("SENSITIVITY SUMMARY")
print("="*70)
print(f"k range [{min(k_values)}, {max(k_values)}]: "
      f"max d_B = {max(k_results['bottleneck_mean']):.3f}")
print(f"r range [{min(r_values):.3f}, {max(r_values):.2f}]: "
      f"max d_B = {max(r_results['bottleneck_mean']):.3f}")
print("="*70 + "\n")

# Plot
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# k sensitivity - bottleneck
ax1.errorbar(k_results['k_values'], k_results['bottleneck_mean'],
             yerr=k_results['bottleneck_std'],
             marker='o', capsize=5, linewidth=2, markersize=8, color='blue')
ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Target')
ax1.axvline(x=k_baseline, color='green', linestyle=':', alpha=0.5, label='Baseline')
ax1.set_xlabel('k (neighbors)', fontsize=12)
ax1.set_ylabel('Bottleneck Distance', fontsize=12)
ax1.set_title('k Sensitivity: Topology', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()

# k sensitivity - survival
ax2.errorbar(k_results['k_values'], 
             [100*m for m in k_results['survival_mean']],
             yerr=[100*s for s in k_results['survival_std']],
             marker='s', capsize=5, linewidth=2, markersize=8, color='green')
ax2.axvline(x=k_baseline, color='green', linestyle=':', alpha=0.5, label='Baseline')
ax2.set_xlabel('k (neighbors)', fontsize=12)
ax2.set_ylabel('Survival Rate (%)', fontsize=12)
ax2.set_title('k Sensitivity: Trimming', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()

# r sensitivity - bottleneck
ax3.errorbar(r_results['r_values'], r_results['bottleneck_mean'],
             yerr=r_results['bottleneck_std'],
             marker='o', capsize=5, linewidth=2, markersize=8, color='purple')
ax3.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Target')
ax3.axvline(x=r_baseline, color='green', linestyle=':', alpha=0.5, label='Baseline')
ax3.set_xlabel('r (opening parameter)', fontsize=12)
ax3.set_ylabel('Bottleneck Distance', fontsize=12)
ax3.set_title('r Sensitivity: Topology', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend()
ax3.set_xscale('log')

# r sensitivity - survival
ax4.errorbar(r_results['r_values'], 
             [100*m for m in r_results['survival_mean']],
             yerr=[100*s for s in r_results['survival_std']],
             marker='s', capsize=5, linewidth=2, markersize=8, color='orange')
ax4.axvline(x=r_baseline, color='green', linestyle=':', alpha=0.5, label='Baseline')
ax4.set_xlabel('r (opening parameter)', fontsize=12)
ax4.set_ylabel('Survival Rate (%)', fontsize=12)
ax4.set_title('r Sensitivity: Trimming', fontsize=13, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.legend()
ax4.set_xscale('log')

plt.tight_layout()
plt.savefig('parameter_sensitivity.png', dpi=300, bbox_inches='tight')
print("Plot saved: parameter_sensitivity.png")

# Save results
with open('parameter_sensitivity_k.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['k', 'd_B_mean', 'd_B_std', 'survival_mean', 'survival_std'])
    for i, k in enumerate(k_results['k_values']):
        writer.writerow([k, k_results['bottleneck_mean'][i], k_results['bottleneck_std'][i],
                        k_results['survival_mean'][i], k_results['survival_std'][i]])

with open('parameter_sensitivity_r.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['r', 'd_B_mean', 'd_B_std', 'survival_mean', 'survival_std'])
    for i, r in enumerate(r_results['r_values']):
        writer.writerow([r, r_results['bottleneck_mean'][i], r_results['bottleneck_std'][i],
                        r_results['survival_mean'][i], r_results['survival_std'][i]])

print("Results saved to: parameter_sensitivity_k.csv, parameter_sensitivity_r.csv\n")
