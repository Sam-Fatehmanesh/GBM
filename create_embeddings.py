#!/usr/bin/env python3
"""
Create PCA and UMAP embeddings from saved next-step truth/prediction H5 files.

Usage:
    python create_embeddings.py /path/to/next_step_truth_pred.h5 [--output-dir /path/to/output]
"""

import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA

try:
    import umap.umap_ as umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: umap-learn not available. Install with: pip install umap-learn")

def create_embeddings(h5_path: str, output_dir: str = None, sample_size: int = 2000, 
                     random_seed: int = 42, umap_n_neighbors: int = 15, 
                     umap_min_dist: float = 0.1, umap_metric: str = 'euclidean'):
    """Create PCA and UMAP 2D embeddings from H5 file."""
    
    h5_path = Path(h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(f"H5 file not found: {h5_path}")
    
    if output_dir is None:
        output_dir = h5_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading data from: {h5_path}")
    print(f"Output directory: {output_dir}")
    
    rng = np.random.RandomState(random_seed)
    
    with h5py.File(h5_path, 'r') as f:
        true_ds = f['true']
        pred_ds = f['pred']
        n = true_ds.shape[0]
        vol_shape = true_ds.shape[1:]
        
        print(f"Found {n} samples with volume shape {vol_shape}")
        
        if n == 0:
            print("No data found in H5 file")
            return
        
        # Sample data
        k = min(sample_size, n)
        if k < n:
            print(f"Sampling {k} out of {n} volumes")
            idx = rng.choice(n, size=k, replace=False)
            idx = np.sort(idx)  # H5 requires sorted indices
        else:
            print(f"Using all {k} volumes")
            idx = np.arange(k)
        
        # Load and flatten
        print("Loading and flattening volumes...")
        X_true = true_ds[idx].reshape(k, -1)
        X_pred = pred_ds[idx].reshape(k, -1)
    
    # Stack and create labels
    X = np.vstack([X_true, X_pred])
    y = np.array([0] * k + [1] * k)  # 0=true, 1=pred
    
    print(f"Combined data shape: {X.shape}")
    print(f"Data range: [{X.min():.4f}, {X.max():.4f}]")
    
    # PCA
    print("Computing PCA...")
    pca = PCA(n_components=2, random_state=random_seed)
    X_pca = pca.fit_transform(X)
    
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], s=8, alpha=0.6, label='True', c='blue')
    plt.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], s=8, alpha=0.6, label='Pred', c='red')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.title('PCA: True vs Predicted Next-Step Volumes')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    pca_path = output_dir / 'pca_true_pred_2d.png'
    plt.savefig(pca_path, dpi=200, bbox_inches='tight')
    print(f"PCA plot saved: {pca_path}")
    plt.close()
    
    # UMAP
    if UMAP_AVAILABLE:
        print("Computing UMAP...")
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            metric=umap_metric,
            random_state=random_seed,
            verbose=True
        )
        X_umap = reducer.fit_transform(X)
        
        plt.figure(figsize=(8, 6))
        plt.scatter(X_umap[y == 0, 0], X_umap[y == 0, 1], s=8, alpha=0.6, label='True', c='blue')
        plt.scatter(X_umap[y == 1, 0], X_umap[y == 1, 1], s=8, alpha=0.6, label='Pred', c='red')
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.title('UMAP: True vs Predicted Next-Step Volumes')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        umap_path = output_dir / 'umap_true_pred_2d.png'
        plt.savefig(umap_path, dpi=200, bbox_inches='tight')
        print(f"UMAP plot saved: {umap_path}")
        plt.close()
    else:
        print("UMAP not available - install umap-learn to enable UMAP plots")

def main():
    parser = argparse.ArgumentParser(description='Create PCA and UMAP embeddings from H5 file')
    parser.add_argument('h5_path', help='Path to next_step_truth_pred.h5 file')
    parser.add_argument('--output-dir', help='Output directory (default: same as H5 file)')
    parser.add_argument('--sample-size', type=int, default=2000, help='Max samples to use (default: 2000)')
    parser.add_argument('--random-seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--umap-neighbors', type=int, default=15, help='UMAP n_neighbors (default: 15)')
    parser.add_argument('--umap-min-dist', type=float, default=0.1, help='UMAP min_dist (default: 0.1)')
    parser.add_argument('--umap-metric', default='euclidean', help='UMAP metric (default: euclidean)')
    
    args = parser.parse_args()
    
    create_embeddings(
        h5_path=args.h5_path,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        random_seed=args.random_seed,
        umap_n_neighbors=args.umap_neighbors,
        umap_min_dist=args.umap_min_dist,
        umap_metric=args.umap_metric
    )

if __name__ == '__main__':
    main()
