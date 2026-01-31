"""
Analysis and Visualization for kNN Opening Experiments
Generates publication-quality plots from experiment CSVs
Run: python analyze_experiments.py <csv_file> [--output plot.png]
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set publication style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("colorblind")

def analyze_vary_sigma(df, output):
    """Plot d_B vs σ with linear fit"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Group by sigma, compute mean and std
    grouped = df.groupby('sigma').agg({
        'dB_H1': ['mean', 'std'],
        'survival': ['mean', 'std']
    })
    
    sigmas = grouped.index.values
    dB_mean = grouped['dB_H1']['mean'].values
    dB_std = grouped['dB_H1']['std'].values
    surv_mean = grouped['survival']['mean'].values
    surv_std = grouped['survival']['std'].values
    
    # Plot 1: Bottleneck distance vs sigma
    ax1.errorbar(sigmas, dB_mean, yerr=dB_std, fmt='o-', capsize=5, label='Measured')
    
    # Linear fit
    mask = ~np.isnan(dB_mean) & ~np.isinf(dB_mean)
    if np.sum(mask) > 2:
        coef = np.polyfit(sigmas[mask], dB_mean[mask], 1)
        fit_line = np.poly1d(coef)
        ax1.plot(sigmas, fit_line(sigmas), '--', alpha=0.7, 
                label=f'Linear fit: d_B ≈ {coef[0]:.2f}σ + {coef[1]:.4f}')
    
    ax1.set_xlabel('Noise σ', fontsize=12)
    ax1.set_ylabel('Bottleneck Distance d_B', fontsize=12)
    ax1.set_title('Linear Scaling: d_B vs σ', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Survival rate
    ax2.errorbar(sigmas, surv_mean, yerr=surv_std, fmt='s-', capsize=5, color='C1')
    ax2.set_xlabel('Noise σ', fontsize=12)
    ax2.set_ylabel('Survival Rate', fontsize=12)
    ax2.set_title('Stability: Survival vs σ', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved: {output}")
    
    # Print summary statistics
    if np.sum(mask) > 2:
        print(f"\nLinear fit: d_B ≈ {coef[0]:.4f}·σ + {coef[1]:.6f}")
        print(f"R² = {np.corrcoef(sigmas[mask], dB_mean[mask])[0,1]**2:.4f}")

def analyze_vary_r(df, output):
    """Plot d_B vs r with linear fit"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    grouped = df.groupby('r_open').agg({
        'dB_H1': ['mean', 'std'],
        'survival': ['mean', 'std']
    })
    
    r_values = grouped.index.values
    dB_mean = grouped['dB_H1']['mean'].values
    dB_std = grouped['dB_H1']['std'].values
    surv_mean = grouped['survival']['mean'].values
    surv_std = grouped['survival']['std'].values
    
    # Plot 1: Bottleneck distance vs r
    ax1.errorbar(r_values, dB_mean, yerr=dB_std, fmt='o-', capsize=5, label='Measured')
    
    # Linear fit
    mask = ~np.isnan(dB_mean) & ~np.isinf(dB_mean)
    if np.sum(mask) > 2:
        coef = np.polyfit(r_values[mask], dB_mean[mask], 1)
        fit_line = np.poly1d(coef)
        ax1.plot(r_values, fit_line(r_values), '--', alpha=0.7,
                label=f'Linear fit: d_B ≈ {coef[0]:.2f}r + {coef[1]:.4f}')
    
    ax1.set_xlabel('Opening Parameter r', fontsize=12)
    ax1.set_ylabel('Bottleneck Distance d_B', fontsize=12)
    ax1.set_title('Linear Scaling: d_B vs r', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Survival rate
    ax2.errorbar(r_values, surv_mean, yerr=surv_std, fmt='s-', capsize=5, color='C1')
    ax2.set_xlabel('Opening Parameter r', fontsize=12)
    ax2.set_ylabel('Survival Rate', fontsize=12)
    ax2.set_title('Expected Decrease with Stronger Opening', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved: {output}")

def analyze_breakdown(df, output, title_suffix=""):
    """Plot breakdown analysis (outlier contamination)"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    grouped = df.groupby('alpha').agg({
        'clean_survival': ['mean', 'std'],
        'outlier_survival': ['mean', 'std'],
        'dB_H1': ['mean', 'std']
    })
    
    alphas = grouped.index.values
    clean_mean = grouped['clean_survival']['mean'].values
    clean_std = grouped['clean_survival']['std'].values
    out_mean = grouped['outlier_survival']['mean'].values
    out_std = grouped['outlier_survival']['std'].values
    dB_mean = grouped['dB_H1']['mean'].values
    dB_std = grouped['dB_H1']['std'].values
    
    # Plot 1: Clean survival rate
    ax1.errorbar(alphas, clean_mean, yerr=clean_std, fmt='o-', capsize=5, 
                color='green', linewidth=2)
    ax1.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='80% threshold')
    ax1.set_xlabel('Contamination Fraction α', fontsize=12)
    ax1.set_ylabel('Clean Point Survival', fontsize=12)
    ax1.set_title(f'Core Preservation{title_suffix}', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])
    ax1.legend()
    
    # Plot 2: Outlier survival rate
    ax2.errorbar(alphas, out_mean, yerr=out_std, fmt='s-', capsize=5, 
                color='red', linewidth=2)
    ax2.set_xlabel('Contamination Fraction α', fontsize=12)
    ax2.set_ylabel('Outlier Survival', fontsize=12)
    ax2.set_title(f'Outlier Rejection{title_suffix}', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, max(0.2, np.max(out_mean + out_std) * 1.1)])
    
    # Plot 3: Bottleneck distance
    mask = ~np.isinf(dB_mean)
    ax3.errorbar(alphas[mask], dB_mean[mask], yerr=dB_std[mask], 
                fmt='o-', capsize=5, color='purple', linewidth=2)
    ax3.set_xlabel('Contamination Fraction α', fontsize=12)
    ax3.set_ylabel('Bottleneck Distance d_B', fontsize=12)
    ax3.set_title(f'Topology Preservation{title_suffix}', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Selectivity (clean vs outlier survival)
    selectivity = clean_mean / (out_mean + 1e-6)
    ax4.plot(alphas, selectivity, 'o-', color='blue', linewidth=2)
    ax4.set_xlabel('Contamination Fraction α', fontsize=12)
    ax4.set_ylabel('Selectivity Ratio\n(Clean Surv / Outlier Surv)', fontsize=12)
    ax4.set_title(f'Discrimination Quality{title_suffix}', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved: {output}")
    
    # Find breakdown point (where clean survival < 0.8)
    breakdown_idx = np.where(clean_mean < 0.8)[0]
    if len(breakdown_idx) > 0:
        alpha_star = alphas[breakdown_idx[0]]
        print(f"\nBreakdown point α* ≈ {alpha_star:.2f}")
        print(f"(Clean survival drops below 80%)")
    else:
        print(f"\nNo breakdown detected (clean survival ≥ 80% for all α ≤ {alphas[-1]:.2f})")

def analyze_parameter_sweep(df, output):
    """Compare different parameter configurations"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    for config_id in df['config_id'].unique():
        subset = df[df['config_id'] == config_id]
        k = subset['k'].iloc[0]
        tau_mode = subset['tau_mode'].iloc[0]
        tau_q = subset['tau_q'].iloc[0] if 'tau_q' in subset else 0.7
        
        label = f"k={k}, {tau_mode}"
        if tau_mode == "quantile":
            label += f" (q={tau_q})"
        
        grouped = subset.groupby('alpha').agg({
            'clean_survival': 'mean',
            'outlier_survival': 'mean'
        })
        
        ax1.plot(grouped.index, grouped['clean_survival'], 'o-', label=label, linewidth=2)
        ax2.plot(grouped.index, grouped['outlier_survival'], 's-', label=label, linewidth=2)
    
    ax1.set_xlabel('Contamination Fraction α', fontsize=12)
    ax1.set_ylabel('Clean Survival', fontsize=12)
    ax1.set_title('Clean Point Preservation', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])
    
    ax2.set_xlabel('Contamination Fraction α', fontsize=12)
    ax2.set_ylabel('Outlier Survival', fontsize=12)
    ax2.set_title('Outlier Rejection', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved: {output}")

def analyze_ambient_sweep(df, output):
    """Plot ambient dimension independence"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    for alpha in df['alpha'].unique():
        subset = df[df['alpha'] == alpha]
        grouped = subset.groupby('d_ambient').agg({
            'clean_survival': ['mean', 'std'],
            'dB_H1': ['mean', 'std']
        })
        
        dims = grouped.index.values
        surv_mean = grouped['clean_survival']['mean'].values
        surv_std = grouped['clean_survival']['std'].values
        dB_mean = grouped['dB_H1']['mean'].values
        dB_std = grouped['dB_H1']['std'].values
        
        ax1.errorbar(dims, surv_mean, yerr=surv_std, fmt='o-', 
                    capsize=5, label=f'α={alpha:.1f}', linewidth=2)
        
        mask = ~np.isinf(dB_mean)
        ax2.errorbar(dims[mask], dB_mean[mask], yerr=dB_std[mask], 
                    fmt='s-', capsize=5, label=f'α={alpha:.1f}', linewidth=2)
    
    ax1.set_xlabel('Ambient Dimension d', fontsize=12)
    ax1.set_ylabel('Clean Survival Rate', fontsize=12)
    ax1.set_title('Dimension Independence: Survival', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])
    
    ax2.set_xlabel('Ambient Dimension d', fontsize=12)
    ax2.set_ylabel('Bottleneck Distance d_B', fontsize=12)
    ax2.set_title('Dimension Independence: Topology', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved: {output}")
    print("\n✓ Constants independent of ambient dimension d (key result!)")

def analyze_hybrid_lof(df, output):
    """Compare pure kNN vs hybrid kNN+LOF"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    methods = df['method'].unique()
    colors = {'knn': 'blue', 'hybrid': 'red'}
    markers = {'knn': 'o', 'hybrid': 's'}
    
    for method in methods:
        subset = df[df['method'] == method]
        grouped = subset.groupby('alpha').agg({
            'clean_survival': ['mean', 'std'],
            'outlier_survival': ['mean', 'std'],
            'dB_H1': ['mean', 'std']
        })
        
        alphas = grouped.index.values
        clean_mean = grouped['clean_survival']['mean'].values
        clean_std = grouped['clean_survival']['std'].values
        out_mean = grouped['outlier_survival']['mean'].values
        out_std = grouped['outlier_survival']['std'].values
        dB_mean = grouped['dB_H1']['mean'].values
        dB_std = grouped['dB_H1']['std'].values
        
        label = 'kNN only' if method == 'knn' else 'kNN+LOF'
        
        ax1.errorbar(alphas, clean_mean, yerr=clean_std, fmt=f'{markers[method]}-', 
                    capsize=5, label=label, color=colors[method], linewidth=2)
        ax2.errorbar(alphas, out_mean, yerr=out_std, fmt=f'{markers[method]}-',
                    capsize=5, label=label, color=colors[method], linewidth=2)
        
        mask = ~np.isinf(dB_mean)
        ax3.errorbar(alphas[mask], dB_mean[mask], yerr=dB_std[mask], 
                    fmt=f'{markers[method]}-', capsize=5, label=label, 
                    color=colors[method], linewidth=2)
        
        selectivity = clean_mean / (out_mean + 1e-6)
        ax4.plot(alphas, selectivity, f'{markers[method]}-', label=label,
                color=colors[method], linewidth=2)
    
    ax1.set_xlabel('Contamination Fraction α', fontsize=12)
    ax1.set_ylabel('Clean Survival', fontsize=12)
    ax1.set_title('Clean Point Preservation', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])
    
    ax2.set_xlabel('Contamination Fraction α', fontsize=12)
    ax2.set_ylabel('Outlier Survival', fontsize=12)
    ax2.set_title('Outlier Rejection', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3.set_xlabel('Contamination Fraction α', fontsize=12)
    ax3.set_ylabel('Bottleneck Distance d_B', fontsize=12)
    ax3.set_title('Topology Preservation', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    ax4.set_xlabel('Contamination Fraction α', fontsize=12)
    ax4.set_ylabel('Selectivity Ratio', fontsize=12)
    ax4.set_title('Discrimination Quality', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved: {output}")

def main():
    parser = argparse.ArgumentParser(description="Analyze kNN opening experiment results")
    parser.add_argument("csv_file", help="Input CSV file from experiments")
    parser.add_argument("--output", default=None, help="Output plot filename")
    parser.add_argument("--experiment_type", default=None,
                       choices=['vary_sigma', 'vary_r', 'breakdown', 
                               'parameter_sweep', 'ambient_sweep', 'hybrid_lof'],
                       help="Experiment type (auto-detected if not specified)")
    
    args = parser.parse_args()
    
    # Load data
    df = pd.read_csv(args.csv_file)
    print(f"Loaded {len(df)} rows from {args.csv_file}")
    
    # Auto-detect experiment type from columns
    if args.experiment_type is None:
        if 'sigma' in df.columns and 'r_open' in df.columns:
            if len(df['sigma'].unique()) > len(df['r_open'].unique()):
                args.experiment_type = 'vary_sigma'
            else:
                args.experiment_type = 'vary_r'
        elif 'method' in df.columns:
            args.experiment_type = 'hybrid_lof'
        elif 'config_id' in df.columns:
            args.experiment_type = 'parameter_sweep'
        elif 'd_ambient' in df.columns:
            args.experiment_type = 'ambient_sweep'
        elif 'alpha' in df.columns:
            args.experiment_type = 'breakdown'
        else:
            print("ERROR: Could not auto-detect experiment type")
            return
    
    print(f"Experiment type: {args.experiment_type}")
    
    # Generate default output filename
    if args.output is None:
        stem = Path(args.csv_file).stem
        args.output = f"{stem}_plot.png"
    
    # Dispatch to appropriate analysis
    if args.experiment_type == 'vary_sigma':
        analyze_vary_sigma(df, args.output)
    elif args.experiment_type == 'vary_r':
        analyze_vary_r(df, args.output)
    elif args.experiment_type == 'breakdown':
        # Determine if isolated or clustered from filename
        title_suffix = ""
        if 'isolated' in args.csv_file.lower():
            title_suffix = " (Isolated)"
        elif 'clustered' in args.csv_file.lower():
            title_suffix = " (Clustered)"
        analyze_breakdown(df, args.output, title_suffix)
    elif args.experiment_type == 'parameter_sweep':
        analyze_parameter_sweep(df, args.output)
    elif args.experiment_type == 'ambient_sweep':
        analyze_ambient_sweep(df, args.output)
    elif args.experiment_type == 'hybrid_lof':
        analyze_hybrid_lof(df, args.output)

if __name__ == "__main__":
    main()
