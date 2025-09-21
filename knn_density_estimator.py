"""
This module implements KNN Density Estimation to select a subset of dataset B
that is most similar to dataset A. It uses FAISS for efficient nearest neighbor search.
"""

import numpy as np
import pandas as pd
import faiss
import os
from scipy.special import loggamma
from sklearn.preprocessing import normalize
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('KNN-Density-Estimator')

def faiss_knn_density(query_vectors, reference_vectors, k, normalize_vectors=True, use_gpu=False):
    """
    KNN density estimation using FAISS.
    """
    
    # Make sure inputs are numpy arrays with float32 type (required by FAISS)
    query_vectors = np.array(query_vectors, dtype=np.float32)
    reference_vectors = np.array(reference_vectors, dtype=np.float32)
    
    d = query_vectors.shape[1]
    
    if normalize_vectors:
        query_vectors = normalize(query_vectors, axis=1)
        reference_vectors = normalize(reference_vectors, axis=1)
    
    n = reference_vectors.shape[0]
    
    if k >= n:
        logger.warning(f"Warning: k ({k}) is >= number of reference points ({n}). Setting k = {n-1}")
        k = n - 1
    
    if use_gpu:
        try:
            res = faiss.StandardGpuResources()
            index = faiss.IndexFlatL2(d)  # L2 distance (Euclidean)
            gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
            gpu_index.add(reference_vectors)
            logger.info("Using GPU FAISS index")
            index_to_use = gpu_index
        except Exception as e:
            logger.warning(f"Failed to use GPU: {e}. Falling back to CPU.")
            index = faiss.IndexFlatL2(d)
            index.add(reference_vectors)
            index_to_use = index
    else:
        index = faiss.IndexFlatL2(d)
        index.add(reference_vectors)
        index_to_use = index
    
    # Search for k+1 neighbors (because FAISS can include the point itself)
    distances, indices = index_to_use.search(query_vectors, k+1)
    
    # Extract k-th distance for each query point
    kth_distances = distances[:, k]
    
    # Define small epsilon to avoid log(0)
    epsilon = 1e-10
    
    # Check for zero or negative values and replace them
    zero_mask = kth_distances <= 0
    if np.any(zero_mask):
        zero_count = np.sum(zero_mask)
        logger.warning(f"Warning: Found {zero_count} zero or negative distance values. Replacing with epsilon={epsilon}")
        kth_distances = np.maximum(kth_distances, epsilon)
    
    # Calculate log of volume of d-dimensional hypersphere
    volumes_log = (d/2) * np.log(np.pi) + d * np.log(kth_distances) - loggamma(d/2 + 1)
    
    # Calculate log of density
    density_log = np.log(k) - np.log(n) - volumes_log
    
    return distances, indices, density_log

def select_high_density_samples(densities, origin_dataset_term, origin_dataset_text, origin_dataset_label, top_k=0.01, is_balanced=False):  
    """
    Select samples with highest density estimates. And get the original text, term and labels,
    """
    
    df = pd.DataFrame({
        'density': densities,
        'term': origin_dataset_term,
        'text': origin_dataset_text,
        'label': origin_dataset_label
    })
    
    # Determine number of samples to select
    if top_k <= 1:
        num_to_select = int(top_k * len(densities))
    else:
        num_to_select = min(int(top_k), len(densities))
    
    if is_balanced:
        num_to_select = round(num_to_select/2)
        df_hate = df[df['label'] == 'hate']
        df_no_hate = df[df['label'] == 'no hate']
        
        df_hate_sorted = df_hate.sort_values(by='density', ascending=False).head(num_to_select)
        df_no_hate_sorted = df_no_hate.sort_values(by='density', ascending=False).head(num_to_select)
        selected_samples = pd.concat([df_hate_sorted, df_no_hate_sorted])
        
    else:
        df_sorted = df.sort_values(by='density', ascending=False)
        selected_samples = df_sorted.head(num_to_select)
    
    return selected_samples

def run_knn_density_estimation(embeddings_a_path, embeddings_b_path, text_b_path, balanced, k=101, percent=0.01, output_path='selected_samples.csv', use_gpu=False):
    """
    Run KNN density estimation 
    """
    
    # Check if input files exist
    for file_path in [embeddings_a_path, embeddings_b_path, text_b_path]:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return None
    
    # Load data
    logger.info(f"Loading dataset A embeddings from {embeddings_a_path}")
    try:
        embeddings_A = np.load(embeddings_a_path)
        logger.info(f"Loaded embeddings A: {embeddings_A.shape}")
    except Exception as e:
        logger.error(f"Error loading dataset A: {e}")
        return None
    
    logger.info(f"Loading dataset B embeddings from {embeddings_b_path}")
    try:
        embeddings_B = np.load(embeddings_b_path)
        logger.info(f"Loaded embeddings B: {embeddings_B.shape}")
    except Exception as e:
        logger.error(f"Error loading dataset B: {e}")
        return None
    
    logger.info(f"Loading dataset B text from {text_b_path}")
    try:
        dataset_B = pd.read_csv(text_b_path)
        logger.info(f"Loaded dataset B: {len(dataset_B)} rows")
    except Exception as e:
        logger.error(f"Error loading dataset B text: {e}")
        return None
    
    # Check if dataset_B has 'text' and 'term' columns
    required_columns = ['text']
    if not all(col in dataset_B.columns for col in required_columns):
        logger.error(f"Dataset B must have a 'text' column. Available columns: {dataset_B.columns.tolist()}")
        return None
    
    # Add 'term' column if not present
    if 'term' not in dataset_B.columns:
        logger.warning("'term' column not found in dataset B. Adding an empty column.")
        dataset_B['term'] = "unknown"
    
    # Check shapes
    if len(embeddings_B) != len(dataset_B):
        logger.error(f"Mismatch in lengths: embeddings_B has {len(embeddings_B)} rows, dataset_B has {len(dataset_B)} rows")
        return None

    # Compute KNN density
    logger.info(f"Computing KNN density with k={k}")
    try:
        distances, indices, densities = faiss_knn_density(
            embeddings_B, embeddings_A, k, normalize_vectors=True, use_gpu=use_gpu
        )
        logger.info("KNN density computation completed")
    except Exception as e:
        logger.error(f"Error computing KNN density: {e}")
        return None
    
    # Select high density samples
    logger.info(f"Selecting top {percent*100:.2f}% samples with highest density")
    result_df = select_high_density_samples(
        densities, origin_dataset_term=dataset_B['term'], origin_dataset_text=dataset_B['text'], origin_dataset_label=dataset_B['label'], top_k=percent, is_balanced=balanced
    )
    logger.info(f"Selected {len(result_df)} samples")
    
    # Save results
    if output_path:
        logger.info(f"Saving results to {output_path}")
        result_df.to_csv(output_path, index=False)
        logger.info("Done!")
    
    return result_df