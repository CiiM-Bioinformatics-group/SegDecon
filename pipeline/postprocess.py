import scanpy as sc
import numpy as np
import pandas as pd
import warnings

# Suppress scanpy warnings
warnings.filterwarnings("ignore")

def load_data(file_path):
    """Load processed AnnData file."""
    return sc.read_h5ad(file_path)

def filter_abundance_data(adata, threshold=0.3):
    """Filter cell abundance data based on a given threshold."""
    abundance_data = adata.obsm['mean_cell_abundance_w_sf']
    filtered_abundance_data = abundance_data.where(abundance_data > threshold, other=0)
    return filtered_abundance_data

def compute_real_cell_counts(adata, filtered_abundance_data):
    """Compute real cell counts based on nuclei count and abundance ratio."""
    abundance_data = adata.obsm['mean_cell_abundance_w_sf']
    total_abundance = abundance_data.sum(axis=1).replace(0, 1e-10)  # Avoid division by zero
    abundance_ratio = abundance_data.div(total_abundance, axis=0)
    
    nuclei_count = adata.obs['nuclei_count'].clip(upper=adata.obs['nuclei_count'].quantile(0.99))
    real_cell_counts = abundance_ratio.mul(nuclei_count, axis=0)
    return real_cell_counts

def update_adata_with_real_counts(adata, real_cell_counts):
    """Update AnnData object with real cell counts."""
    real_cell_counts.columns = real_cell_counts.columns.str.replace(
        r'^meanscell_abundance_w_sf_', 'real_', regex=True
    )
    for col in real_cell_counts.columns:
        adata.obs[col] = real_cell_counts[col]
    adata.obsm['real_cell_counts'] = real_cell_counts

def save_processed_data(adata, output_path):
    """Save the updated AnnData object."""
    adata.write(output_path)
    print(f"Processed data saved as '{output_path}'.")

def run_pipeline():
    """Run the postprocessing pipeline."""
    print("Loading data...")
    adata = load_data("../data/deconvolution_output.h5ad")
    
    print("Filtering abundance data...")
    filtered_abundance_data = filter_abundance_data(adata)
    
    print("Computing real cell counts...")
    real_cell_counts = compute_real_cell_counts(adata, filtered_abundance_data)
    
    print("Updating AnnData object...")
    update_adata_with_real_counts(adata, real_cell_counts)
    
    print("Saving processed data...")
    save_processed_data(adata, "../data/segdecon_results.h5ad")
    
    print("Postprocessing pipeline completed.")

if __name__ == "__main__":
    run_pipeline()
