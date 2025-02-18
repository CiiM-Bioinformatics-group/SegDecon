#!/usr/bin/env python
# coding: utf-8

import scanpy as sc
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
import json
import matplotlib.pyplot as plt
from PIL import Image
import os

import warnings
warnings.filterwarnings('ignore')

# Here we use data with bin=16um to maximally mimic Visium data.

# Load Visium HD data
adata = sc.read_10x_h5('../binned_outputs/square_016um/filtered_feature_bc_matrix.h5')

# Calculate QC metrics
adata.var_names_make_unique()
adata.var["mt"] = adata.var_names.str.startswith("mt-")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

# Load the Spatial Coordinates
tissue_position_file = '../binned_outputs/square_016um/spatial/tissue_positions.parquet'
df_tissue_positions = pd.read_parquet(tissue_position_file)

# Set the index of the dataframe to the barcodes
df_tissue_positions = df_tissue_positions.set_index('barcode')

# Create an index in the dataframe to check joins
df_tissue_positions['index'] = df_tissue_positions.index

# Add the tissue positions to the metadata (adata.obs)
adata.obs = pd.merge(adata.obs, df_tissue_positions, left_index=True, right_index=True)

# Create a GeoDataFrame from the DataFrame of coordinates
geometry = [Point(xy) for xy in zip(df_tissue_positions['pxl_col_in_fullres'], df_tissue_positions['pxl_row_in_fullres'])]
gdf_coordinates = gpd.GeoDataFrame(df_tissue_positions, geometry=geometry)

# Filter tissue_positions to match adata.obs index
df_tissue_positions = df_tissue_positions.loc[adata.obs.index]

# Ensure the number of entries match
assert df_tissue_positions.shape[0] == adata.obs.shape[0], "Row count mismatch, please check index alignment"

# Store pixel coordinates (full resolution) into a 2D array
spatial_coords = np.array([df_tissue_positions['pxl_col_in_fullres'], 
                           df_tissue_positions['pxl_row_in_fullres']]).T

# Store spatial coordinates into adata.obsm
adata.obsm['spatial'] = spatial_coords

# Read JSON file (assumed to contain image scale information, etc.)
scalefactors_file = '../binned_outputs/square_016um/spatial/scalefactors_json.json'
with open(scalefactors_file, 'r') as f:
    scalefactors = json.load(f)

# Read image files (assumed to have hires and lowres images)
hires_image_file = '../binned_outputs/square_016um/spatial/tissue_hires_image.png'
lowres_image_file = '../binned_outputs/square_016um/spatial/tissue_lowres_image.png'

# Use PIL to read images and convert them to NumPy arrays
img_hires = np.array(Image.open(hires_image_file))
img_lowres = np.array(Image.open(lowres_image_file))

# Add data to adata.uns
adata.uns['spatial'] = {
    'VisiumHD_Mouse_Brain': {
        'images': {
            'hires': img_hires,
            'lowres': img_lowres
        },
        'scalefactors': scalefactors
    }
}

adata

# 1. From the Visium HD data's scalefactors file, we know microns_per_pixel is 0.2738242950835738.
#    We want the distance between each spot to be 100 microns, so the corresponding pixel distance is 100 / 0.2738242950835738 = 1000 / 2.738.
# 2. Because the spot is a circle but we want it to capture all spatial gene information, the diameter needs to be larger than 100 microns,
#    specifically 50√2 microns = 258 pixels. However, here we allow some gene information loss, so we use 1000 / 2.738 / 2 = 183 pixels.

from scipy.spatial import cKDTree
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
import scanpy as sc
import json
from PIL import Image

# Set the background image boundaries
max_coordinate = 25000

# Create the background plot
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(0, max_coordinate)
ax.set_ylim(0, max_coordinate)

# Draw background grid lines
x_ticks = np.arange(0, max_coordinate + 1000, 1000)
y_ticks = np.arange(0, max_coordinate + 1000, 1000)
ax.set_xticks(x_ticks)
ax.set_yticks(y_ticks)
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
ax.invert_yaxis()

if 'spatial' in adata.obsm:
    spots_coords = adata.obsm['spatial']
    ax.scatter(spots_coords[:, 0], spots_coords[:, 1], s=5, color='blue', alpha=0.6)

tree = cKDTree(spots_coords)
background_x = np.arange(0, max_coordinate + 1000 / 2.738, 1000 / 2.738)
background_y = np.arange(0, max_coordinate + 1000 / 2.738, 1000 / 2.738)
background_coords = np.array([[x, y] for x in background_x for y in background_y])

def find_neighbors(coord):
    neighbors_idx = tree.query_ball_point(coord, r=183)
    return (coord, neighbors_idx, len(neighbors_idx)) if len(neighbors_idx) > 0 else None

def process_parallel(coords, tree):
    results = []
    with Pool(processes=cpu_count()) as pool:
        for result in tqdm(pool.imap(find_neighbors, coords), total=len(coords), desc='Finding Neighbors'):
            results.append(result)
    return results

results = process_parallel(background_coords, tree)
merged_indices = {}
merged_data = []
new_spatial_coords = []
new_obs = []

merged_mask = np.zeros(spots_coords.shape[0], dtype=bool)  # Prevent duplicate merging

# Generate merged_indices
for i, result in enumerate(results):
    if result is not None:
        coord, neighbors_idx, neighbor_count = result
        neighbors_idx = np.array(neighbors_idx)

        if merged_mask[neighbors_idx].any():
            continue
        merged_mask[neighbors_idx] = True

        # Save merged indices
        merged_indices[i] = neighbors_idx.tolist()

        # Calculate merged expression data (using sum) for multiple spots
        if len(neighbors_idx) > 1:
            merged_spot_data = np.sum(adata.X[neighbors_idx, :].toarray(), axis=0)
        else:
            merged_spot_data = adata.X[neighbors_idx[0], :].toarray().flatten()

        # Save the merged expression data as a flattened 1D array
        merged_data.append(merged_spot_data.flatten())

        # Save new spatial coordinates
        new_spatial_coords.append(coord)

# Convert merged expression data into a 2D matrix
merged_expression_matrix = np.array(merged_data)

# Remove unnecessary dimensions using np.squeeze()
merged_expression_matrix = np.squeeze(merged_expression_matrix)

# Ensure merged_expression_matrix is a numpy array
merged_expression_matrix = np.array(merged_data)  # Ensure it is a numpy array, not a list

import numpy as np
import pandas as pd
import scanpy as sc

# Assuming merged_expression_matrix and merged_indices have been correctly assigned

# Updated processing code:
new_obs = []

# Ensure each index in merged_indices is valid in the merged expression matrix
for indices in merged_indices.values():
    # Ensure indices do not exceed the range of the merged matrix
    indices = [idx for idx in indices if idx < len(merged_expression_matrix)]
    
    if len(indices) > 0:  # Ensure indices are not empty
        # Calculate merged total_counts
        new_obs.append({
            'total_counts': np.sum(merged_expression_matrix[indices, :], axis=0).sum()  # Compute merged total_counts
        })
    else:
        new_obs.append({
            'total_counts': 0  # If no valid indices, set total_counts to 0
        })

# Convert new_obs to a DataFrame
new_obs_df = pd.DataFrame(new_obs)

# Ensure new_spatial_coords is a numpy array
new_spatial_coords = np.array(new_spatial_coords)  # Ensure new_spatial_coords is a numpy array, not a list

# Create a new AnnData object
new_adata = sc.AnnData(X=merged_expression_matrix, obs=new_obs_df, var=adata.var.copy(), obsm={'spatial': new_spatial_coords})

# Print the total sum of merged data
total_counts = np.sum(merged_expression_matrix, axis=1)
print("Total merged total_counts:", total_counts.sum())

import numpy as np
import json
from PIL import Image

# Update the uns information, including images and scale factors
new_adata.uns = adata.uns.copy()

# Update QC information, such as total counts
merged_data = np.array(merged_data)  # Ensure merged_data is a NumPy array
print("Merged data shape:", merged_data.shape)  # Check the shape of merged data
new_adata.obs['total_counts'] = np.sum(merged_data, axis=1)

# Ensure 'mt' is a boolean array for selecting mitochondrial genes
if 'mt' in adata.var:
    mt_gene_mask = adata.var['mt'].values  # Convert 'mt' to a NumPy array
    new_adata.obs['total_counts_mt'] = np.sum(merged_data[:, mt_gene_mask], axis=1)

# Copy other QC information
obs_columns_to_copy = ['n_genes_by_counts', 'log1p_n_genes_by_counts', 
                       'log1p_total_counts', 'pct_counts_in_top_50_genes',
                       'pct_counts_in_top_100_genes', 'pct_counts_in_top_200_genes',
                       'pct_counts_in_top_500_genes', 'pct_counts_mt', 
                       'in_tissue', 'array_row', 'array_col']

# Ensure new_adata.obs columns match obs_columns_to_copy
for col in obs_columns_to_copy:
    if col in adata.obs.columns:  # Ensure the column exists in original data
        new_adata.obs[col] = adata.obs[col].values[:new_adata.n_obs]  # Ensure length matches
    else:
        print(f"Warning: {col} not found in adata.obs.")

# Check if merged_indices is not empty
if merged_indices:
    new_adata.obs['pxl_row_in_fullres'] = [
        np.mean(adata.obs['pxl_row_in_fullres'].iloc[indices]) if len(indices) > 0 else np.nan
        for indices in merged_indices.values()
    ][:new_adata.n_obs]  # Ensure length matches
    new_adata.obs['pxl_col_in_fullres'] = [
        np.mean(adata.obs['pxl_col_in_fullres'].iloc[indices]) if len(indices) > 0 else np.nan
        for indices in merged_indices.values()
    ][:new_adata.n_obs]  # Ensure length matches
else:
    # If no merged indices, assign default values (e.g., np.nan or other appropriate values)
    new_adata.obs['pxl_row_in_fullres'] = np.full(new_adata.n_obs, np.nan)
    new_adata.obs['pxl_col_in_fullres'] = np.full(new_adata.n_obs, np.nan)

# Read and update scalefactors and image information
scalefactors_file = '../binned_outputs/square_016um/spatial/scalefactors_json.json'
with open(scalefactors_file, 'r') as f:
    scalefactors = json.load(f)

# Update bin size and spot diameter
new_bin_size_um = 55  # Updated bin size in microns
new_spot_diameter_fullres = 300  # Updated spot diameter in full resolution

'''[ spot_diameter_fullres ] = (Actual physical size in microns) / (microns_per_pixel)

Setting actual physical size to 55 microns, microns_per_pixel is 0.2738242950835738:

[ spot_diameter_fullres ] = 55 / 0.2738242950835738 = 200.96 pixels

We let new_spot_diameter_fullres = 1.5 times 200.96, so that's better for collecting cell counts in the next step.
But the spot diameter in full resolution here does not store any gene information; it's only for visualization.
'''

# Update scale factors
scalefactors['spot_diameter_fullres'] = new_spot_diameter_fullres
scalefactors['bin_size_um'] = new_bin_size_um

# Read high-resolution and low-resolution images
hires_image_file = '../binned_outputs/square_016um/spatial/tissue_hires_image.png'
lowres_image_file = '../binned_outputs/square_016um/spatial/tissue_lowres_image.png'

# Use PIL to read images and convert them to NumPy arrays
img_hires = np.array(Image.open(hires_image_file))
img_lowres = np.array(Image.open(lowres_image_file))

# Add updated images and scale factors to new_adata.uns
new_adata.uns['spatial'] = {
    'VisiumHD_Mouse_Brain': {
        'images': {
            'hires': img_hires,   # High-resolution image
            'lowres': img_lowres  # Low-resolution image
        },
        'scalefactors': scalefactors  # Updated scale factors
    }
}

# Check the structure of new_adata
print(new_adata)
print(new_adata.obs['total_counts'].describe())

print(new_adata.obs['pxl_row_in_fullres'].head(10))

# Print the total sum of merged total_counts
total_counts = np.sum(merged_expression_matrix, axis=1)
print("Total merged total_counts:", total_counts.sum())

new_adata.write('../MOUSE BRAIN/anndatas/synthetic_visiumdata_uncounted.h5ad')

plt.rcParams["figure.figsize"] = (8, 8)
sc.pl.spatial(adata, img_key="hires", color=["total_counts"], cmap='turbo')

plt.rcParams["figure.figsize"] = (8, 8)
sc.pl.spatial(new_adata, img_key="hires", color=["total_counts"], cmap='turbo')

# Create subplots for spatial plots
fig, axs = plt.subplots(1, 2, figsize=(15, 7))

# Use sc.pl.spatial to display clustering information
sc.pl.spatial(
    adata, 
    img_key="hires", 
    color="total_counts",
    cmap='turbo', 
    alpha=1, 
    # frameon=False, 
    show=False, 
    ax=axs[0]
)
axs[0].set_title('Total Gene Expression per Spot', fontsize=16)  # Set title font size
axs[0].tick_params(axis='both', labelsize=15)  # Set axis label font size

# Visualize total gene expression per spot
sc.pl.spatial(
    new_adata, 
    img_key="hires", 
    color="total_counts",
    cmap='turbo', 
    alpha=1,
    # frameon=False, 
    show=False, 
    ax=axs[1]
)
axs[1].set_title('Total Gene Expression per Spot simulated', fontsize=16)  # Set title font size
axs[1].tick_params(axis='both', labelsize=15)  # Set axis label font size

# Show the plot
plt.tight_layout()
plt.show()

# Perform basic filtering on new_adata
sc.pp.filter_cells(new_adata, min_counts=0)
sc.pp.filter_cells(new_adata, max_counts=140000)
new_adata = new_adata[new_adata.obs["pct_counts_mt"] < 20]
sc.pp.filter_genes(new_adata, min_cells=10)

# Perform normalization
sc.pp.normalize_total(new_adata, inplace=True)
sc.pp.log1p(new_adata)
# Determine the top 3000 highly variable genes.
sc.pp.highly_variable_genes(new_adata, n_top_genes=3000)

# Dimensionality reduction and clustering
sc.pp.pca(new_adata)
sc.pp.neighbors(new_adata)
sc.tl.umap(new_adata)

sc.tl.leiden(new_adata, key_added="clusters")

plt.rcParams["figure.figsize"] = (8, 8)
sc.pl.spatial(new_adata, img_key="hires", color=["clusters"])

plt.rcParams["figure.figsize"] = (8, 8)
sc.pl.umap(new_adata, color=["clusters"])

import matplotlib.pyplot as plt
import seaborn as sns

sns.histplot(adata.obs['total_counts'], bins=100)
plt.xlabel('Total UMI Counts per Cell')
plt.ylabel('Number of Cells')
plt.show()

# Perform basic filtering on adata
sc.pp.filter_cells(adata, min_counts=200)
sc.pp.filter_cells(adata, max_counts=4000)
adata = adata[adata.obs["pct_counts_mt"] < 20]
sc.pp.filter_genes(adata, min_cells=10)

# Perform normalization
sc.pp.normalize_total(adata, inplace=True)
sc.pp.log1p(adata)
# Determine the top 5000 highly variable genes.
sc.pp.highly_variable_genes(adata, n_top_genes=5000)

# Dimensionality reduction and clustering
sc.pp.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=1.5, key_added="clusters")

plt.rcParams["figure.figsize"] = (4,4)
sc.pl.umap(adata, color=["clusters"])

# Create subplots for UMAP plots
fig, axs = plt.subplots(1, 2, figsize=(9, 4))

# Use sc.pl.umap to display clustering information
sc.pl.umap(
    adata, 
    color="clusters", 
    alpha=1, 
    # frameon=False, 
    show=False, 
    ax=axs[0]
)
axs[0].set_title('Clusters of Visium HD', fontsize=14)  # Set title font size
axs[0].tick_params(axis='both', labelsize=12)  # Set axis label font size

# Visualize clusters on synthetic ST data
sc.pl.umap(
    new_adata, 
    color="clusters", 
    alpha=1, 
    # frameon=False, 
    show=False, 
    ax=axs[1]
)
axs[1].set_title('Clusters of Synthetic ST data', fontsize=14)  # Set title font size
axs[1].tick_params(axis='both', labelsize=12)  # Set axis label font size

# Show the plot
plt.tight_layout()
plt.show()
