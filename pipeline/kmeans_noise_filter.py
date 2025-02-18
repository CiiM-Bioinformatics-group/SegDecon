import numpy as np
import cv2
from sklearn.cluster import KMeans

def get_default_noise_thresholds():
    """
    Returns default HSV noise thresholds based on prior knowledge.
    This method is the primary approach for noise masking.
    """
    
    lower_noise = np.array([154, 137, 107], dtype=np.uint8)  # Default lower bound
    upper_noise = np.array([162, 207, 211], dtype=np.uint8)  # Default upper bound
    return lower_noise, upper_noise

def get_kmeans_noise_thresholds(img_hsv, crop_region=None, n_clusters=3):
    """
    Provides a reference HSV noise threshold using KMeans clustering.
    This method does NOT replace the default threshold but can serve as an alternative.

    Parameters:
    - img_hsv: HSV image (numpy array).
    - crop_region: Tuple (y1, y2, x1, x2), optional. Crops the region before analysis.
    - n_clusters: Number of clusters for KMeans (default: 3).

    Returns:
    - lower_noise: Computed lower bound HSV.
    - upper_noise: Computed upper bound HSV.
    """
    if crop_region:
        y1, y2, x1, x2 = crop_region
        img_hsv = img_hsv[y1:y2, x1:x2]

    h_channel = img_hsv[:, :, 0]
    s_channel = img_hsv[:, :, 1]
    v_channel = img_hsv[:, :, 2]
    h_flat = h_channel.flatten().reshape(-1, 1)

    # Run KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(h_flat)
    labels = kmeans.labels_.reshape(h_channel.shape)

    # Identify clusters corresponding to different colors
    sorted_clusters = np.argsort(kmeans.cluster_centers_.flatten())
    blue_cluster = sorted_clusters[0]  # Dark blue (assumed lowest H value)
    purple_cluster = sorted_clusters[1]  # Dark purple (middle H value)
    pink_cluster = sorted_clusters[2]  # Pink (highest H value)

    # Extract pixels from the identified noise cluster (purple)
    noise_pixels = np.column_stack(np.where(labels == purple_cluster))
    if len(noise_pixels) == 0:
        return get_default_noise_thresholds()
    selected_indices = np.random.choice(len(noise_pixels), min(8, len(noise_pixels)), replace=False)
    selected_hsv = img_hsv[noise_pixels[selected_indices, 0], noise_pixels[selected_indices, 1]]

    # Compute threshold
    lower_noise = np.min(selected_hsv, axis=0).astype(np.uint8)
    upper_noise = np.max(selected_hsv, axis=0).astype(np.uint8)

    return lower_noise, upper_noise
