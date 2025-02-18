import os
import logging
import numpy as np
import scanpy as sc

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def setup_logger(name):
    """Create and return a logger with the given name."""
    logger = logging.getLogger(name)
    return logger

# File management utilities
def ensure_directory(path):
    """Ensure the directory exists, create if not."""
    if not os.path.exists(path):
        os.makedirs(path)

def load_anndata(file_path):
    """Load an AnnData object safely."""
    if os.path.exists(file_path):
        return sc.read_h5ad(file_path)
    else:
        raise FileNotFoundError(f"File not found: {file_path}")

def save_anndata(adata, file_path):
    """Save an AnnData object to file."""
    ensure_directory(os.path.dirname(file_path))
    adata.write(file_path)
    logging.info(f"Saved AnnData to {file_path}")

# Data preprocessing utilities
def filter_genes(adata, min_counts=10, min_cells=3):
    """Filter genes based on minimum count and cell expression."""
    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.filter_genes(adata, min_counts=min_counts)
    return adata

def normalize_data(adata):
    """Normalize gene expression data."""
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata

def preprocess_annotations(adata):
    """Preprocess annotations (e.g., removing unwanted genes or categories)."""
    if 'annotation' in adata.obs:
        adata.obs['annotation'] = adata.obs['annotation'].str.replace("unwanted_category", "")
    return adata

# Spatial transcriptomics utilities
def extract_coordinates(adata):
    """Extract spatial coordinates from AnnData object."""
    return adata.obsm['spatial'] if 'spatial' in adata.obsm else None
