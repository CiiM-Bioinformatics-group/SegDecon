import os
import sys
import scanpy as sc
import numpy as np
import pandas as pd
import cell2location as c2l
import matplotlib.pyplot as plt
import warnings

# Set parameters
def setup_environment():
    """Setup the environment by installing required packages"""
    IN_COLAB = "google.colab" in sys.modules
    if IN_COLAB:
        get_ipython().system('pip install --quiet scvi-colab')
        from scvi_colab import install
        install()
        get_ipython().system('pip install --quiet git+https://github.com/BayraktarLab/cell2location#egg=cell2location[tutorials]')

def load_data():
    """Load the datasets"""
    # Note: The following data is specific to the mouse brain.
    # If working with tissues other than mouse brain, users will need to generate their own reference snRNA data.
    
    # Define paths
    data_folder = './data'  # Path where data is stored (both tissue-specific and reference data)
    reg_path = f'{data_folder}/mouse_brain_snrna/regression_model'  # Path for reference data
    
    # Check if the reference data exists, if not, download it
    if not os.path.exists(reg_path):
        os.makedirs(reg_path, exist_ok=True)  # Create necessary folders for reference data
        print("Downloading the reference data from Cell2Location...")
        
        # Download the reference data (snRNA-seq data with signatures of reference cell types for mouse brain)
        # Data is from Cell2Location, tissue is from mouse brain, and contains signatures for various cell types
        os.system(f'cd {reg_path} && wget https://cell2location.cog.sanger.ac.uk/tutorial/mouse_brain_snrna/regression_model/RegressionGeneBackgroundCoverageTorch_65covariates_40532cells_12819genes/sc.h5ad')

    # Load the reference and tissue-specific datasets
    print("Loading the datasets...")
    
    # Spatial transcription data after nuclei segmentation and counting,for deconvolution (input data)
    adata_st = sc.read_h5ad(f'{data_folder}/deconvolution_input.h5ad')
    
    # Reference data: snRNA-seq data with signatures of reference cell types
    # Data is from the Cell2Location tutorial for mouse brain
    adata_ref = sc.read(f'{reg_path}/RegressionGeneBackgroundCoverageTorch_65covariates_40532cells_12819genes/sc.h5ad')
    
    return adata_st, adata_ref


def preprocess_data(adata_st, adata_ref):
    """Preprocess the data"""
    # Prepare feature names
    adata_st.var["feature_name"] = adata_st.var_names
    adata_st.var.set_index("gene_ids", drop=True, inplace=True)
    
    # Handle mitochondria-encoded genes
    adata_st.var['MT_gene'] = [gene.startswith('mt-') for gene in adata_st.var.index]
    adata_st.obsm['MT'] = adata_st[:, adata_st.var['MT_gene'].values].X.toarray()
    adata_st = adata_st[:, ~adata_st.var['MT_gene'].values]

    return adata_st, adata_ref

def get_shared_features(adata_st, adata_ref):
    """Find and align shared features between spatial and reference data"""
    shared_features = [feature for feature in adata_st.var_names if feature in adata_ref.var_names]
    adata_ref = adata_ref[:, shared_features].copy()
    adata_st = adata_st[:, shared_features].copy()
    return adata_st, adata_ref

def prepare_for_cell2location(adata_st, adata_ref):
    """Prepare data for Cell2location"""
    covariate_col_names = 'annotation_1'
    inf_aver = adata_ref.raw.var.copy()
    inf_aver = inf_aver.loc[:, [f'means_cov_effect_{covariate_col_names}_{i}' for i in adata_ref.obs[covariate_col_names].unique()]]
    inf_aver.columns = [sub(f'means_cov_effect_{covariate_col_names}_{i}', '', i) for i in adata_ref.obs[covariate_col_names].unique()]
    inf_aver = inf_aver.iloc[:, inf_aver.columns.argsort()]
    inf_aver = inf_aver * adata_ref.uns['regression_mod']['post_sample_means']['sample_scaling'].mean()

    # Calculate mean and variance for nuclei_count
    mean_cells = adata_st.obs['nuclei_count'].mean().astype('float32')
    var_cells = adata_st.obs['nuclei_count'].var().astype('float32')
    
    # Calculate N_cells_mean_var_ratio
    N_cells_mean_var_ratio = mean_cells / var_cells
    
    # Find shared genes and subset both anndata and reference signatures
    intersect = np.intersect1d(adata_st.var_names, inf_aver.index)
    adata_st = adata_st[:, intersect].copy()
    inf_aver = inf_aver.loc[intersect, :].copy()
    
    return inf_aver, mean_cells, var_cells, N_cells_mean_var_ratio, adata_st

def setup_cell2location_model(adata_st, inf_aver):
    """Setup and initialize the Cell2location model"""
    c2l.models.Cell2location.setup_anndata(adata=adata_st)
    model = c2l.models.Cell2location(
        adata_st,
        cell_state_df=inf_aver,
        N_cells_per_location=float(adata_st.obs['nuclei_count'].mean()),
        N_cells_mean_var_ratio=float(adata_st.obs['nuclei_count'].var())
    )
    return model

def train_and_export(model, adata_st):
    """Train model and export results"""
    # Train the model
    model.train(max_epochs=30000, batch_size=None, train_size=1)
    
    # Plot training history
    model.plot_history()
    plt.legend(labels=['full data training'])

    # Export posterior
    adata_st = model.export_posterior(
        adata_st,
        sample_kwargs={
            "num_samples": 1000,
            "batch_size": model.adata.n_obs,
        },
    )

    # Plot QC
    model.plot_QC()
    
    return adata_st

def save_results(adata_st):
    """Save results"""
    # Update the adata_st.obs with the abundance data
    adata_st.obs[adata_st.uns["mod"]["factor_names"]] = adata_st.obsm["q05_cell_abundance_w_sf"]

    # Save the results to a .h5ad file
    adata_st.write("../data/deconvolution_output.h5ad")
    
    # Save additional output as CSV
    # pd.DataFrame(adata_st.obsm['q05_cell_abundance_w_sf']).to_csv("../data/deconvolution_output.csv")
    
def run_deconvolution_pipeline():
    """Run the deconvolution pipeline"""
    print("Loading data...")
    adata_st, adata_ref = load_data()

    print("Preprocessing data...")
    adata_st, inf_aver, mean_cells, var_cells = preprocess_data(adata_st, adata_ref)

    print("Setting up Cell2Location model...")
    model = setup_cell2location_model(adata_st, inf_aver, mean_cells, var_cells)

    print("Training model and exporting results...")
    adata_st = train_and_export(model, adata_st)

    print("Saving results...")
    save_results(adata_st)

if __name__ == "__main__":
    run_deconvolution_pipeline()
