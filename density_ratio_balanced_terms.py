"""
Density Ratio Sampler with Term Balancing

This script implements KNN Density Estimation using FAISS to create a balanced dataset
that follows the distribution of the Russian dataset. Unlike the previous implementation,
this script ensures that samples are drawn from all terms in a more balanced way.
"""

import os
import sys
import numpy as np
import pandas as pd
import faiss
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import loggamma
from sklearn.preprocessing import normalize
from tqdm import tqdm
from dataHandler import getToxigenDatasetListClass, getCompleteRussDataset, getListOfIdTerms

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('Density-Ratio-Balanced-Sampler')

# Set random seed for reproducibility
np.random.seed(42)

def faiss_knn_density(query_vectors, reference_vectors, k, normalize_vectors=True, use_gpu=True):
    """
    KNN density estimation using FAISS.
    
    Parameters:
    -----------
    query_vectors: points where we want to estimate density
    reference_vectors: dataset for density estimation
    k: number of neighbors
    normalize_vectors: whether to normalize vectors before computation
    use_gpu: whether to use GPU acceleration for FAISS
    
    Returns:
    --------
    distances: distances to k nearest neighbors
    density_log: log of estimated density at each query point
    """
    # Make sure inputs are numpy arrays with float32 type (required by FAISS)
    query_vectors = np.array(query_vectors, dtype=np.float32)
    reference_vectors = np.array(reference_vectors, dtype=np.float32)
    
    d = query_vectors.shape[1]
    
    if normalize_vectors:
        query_vectors = normalize(query_vectors, axis=1)
        reference_vectors = normalize(reference_vectors, axis=1)
    
    n = reference_vectors.shape[0]
    
    # Make sure k is not larger than the number of reference points
    if k >= n:
        logger.warning(f"Warning: k ({k}) is >= number of reference points ({n}). Setting k = {n-1}")
        k = n - 1
    
    # Use GPU if available and requested
    if use_gpu:
        try:
            res = faiss.StandardGpuResources()
            index = faiss.IndexFlatL2(d)
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
    distances, _ = index_to_use.search(query_vectors, k+1)
    
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
    
    return distances, density_log, volumes_log

def load_datasets_and_embeddings():
    """
    Load the datasets and embeddings needed for density estimation.
    
    Returns:
    --------
    russian_embeddings: embeddings of Russian dataset
    term_embeddings_dict: dictionary mapping terms to their embeddings
    term_dataset_dict: dictionary mapping terms to their dataframes
    """
    # Hardcoded paths
    russian_embeddings_path = 'term_embeddings/embeddings_russian.npy'
    
    # Load Russian embeddings
    logger.info(f"Loading Russian embeddings from {russian_embeddings_path}")
    try:
        russian_embeddings = np.load(russian_embeddings_path)
        logger.info(f"Loaded Russian embeddings: {russian_embeddings.shape}")
    except Exception as e:
        logger.error(f"Error loading Russian embeddings: {e}")
        sys.exit(1)
    
    # Load dataset with term information
    logger.info("Loading dataset with term information...")
    try:
        df_with_terms = getToxigenDatasetListClass(is_random=False)
        logger.info(f"Loaded dataset: {len(df_with_terms)} samples")
    except Exception as e:
        logger.error(f"Error loading dataset with terms: {e}")
        sys.exit(1)
    
    # Load embeddings for each term
    term_embeddings_dict = {}
    term_dataset_dict = {}
    
    # Get list of all terms
    term_list = getListOfIdTerms()[:-1]  # Exclude Russian
    
    for term in term_list:
        term_embeddings_path = f'term_embeddings/embeddings_{term}.npy'
        
        try:
            term_embeddings = np.load(term_embeddings_path)
            term_embeddings_dict[term] = term_embeddings
            
            # Get the subset of the dataset for this term
            term_df = df_with_terms[df_with_terms['term'] == term]
            term_dataset_dict[term] = term_df
            
            logger.info(f"Loaded embeddings and data for term '{term}': {term_embeddings.shape}, {len(term_df)} samples")
        except Exception as e:
            logger.warning(f"Error loading embeddings for term '{term}': {e}")
    
    return russian_embeddings, term_embeddings_dict, term_dataset_dict

def calculate_term_density_ratios(term_embeddings_dict, term_dataset_dict, russian_embeddings, k=1000):
    """
    Calculate density ratios for each term's embeddings with respect to Russian embeddings
    and other term embeddings.
    
    Parameters:
    -----------
    term_embeddings_dict: dictionary mapping terms to their embeddings
    term_dataset_dict: dictionary mapping terms to their dataframes
    russian_embeddings: embeddings of Russian dataset
    k: number of neighbors for KNN density estimation
    
    Returns:
    --------
    term_density_ratios_dict: dictionary mapping terms to their density ratios
    """
    term_density_ratios_dict = {}
    
    for term, embeddings in term_embeddings_dict.items():
        logger.info(f"Calculating density ratios for term '{term}'...")
        
        # Calculate density to Russian dataset
        _, density_to_russian, _ = faiss_knn_density(
            embeddings,
            russian_embeddings,
            k,
            normalize_vectors=True,
            use_gpu=True
        )
        
        # Calculate densities to all other terms, excluding the current term
        all_other_densities = []
        for other_term, other_embeddings in term_embeddings_dict.items():
            if other_term != term:
                _, density_to_other_term, _ = faiss_knn_density(
                    embeddings,
                    other_embeddings,
                    k,
                    normalize_vectors=True,
                    use_gpu=True
                )
                all_other_densities.append(density_to_other_term)
        
        # Stack and average the densities to other terms
        if all_other_densities:
            stacked_densities = np.stack(all_other_densities, axis=0)
            avg_density_to_others = np.mean(stacked_densities, axis=0)
            
            # Calculate log ratio (log(density_russian) - log(density_others))
            # Since we're already in log space, subtraction equals division in linear space
            log_density_ratios = density_to_russian - avg_density_to_others
        else:
            # If there are no other terms, just use the Russian density directly
            logger.warning(f"No other term embeddings found for '{term}'. Using only Russian density.")
            log_density_ratios = density_to_russian
        
        # Store density ratios
        term_density_ratios_dict[term] = log_density_ratios
        
    return term_density_ratios_dict

def create_balanced_sampled_dataset(term_density_ratios_dict, term_dataset_dict, total_samples=1000000, min_samples_per_term=1000, max_repetitions=10):
    """
    Create a balanced dataset with samples from each term, weighted by their density ratios.
    
    Parameters:
    -----------
    term_density_ratios_dict: dictionary mapping terms to their density ratios
    term_dataset_dict: dictionary mapping terms to their dataframes
    total_samples: total number of samples in the output dataset
    min_samples_per_term: minimum number of samples to include from each term
    max_repetitions: maximum number of times a single sample can be repeated
    
    Returns:
    --------
    sampled_df: balanced dataset with samples from all terms
    """
    all_terms = list(term_density_ratios_dict.keys())
    num_terms = len(all_terms)
    
    # First, ensure we have a minimum number of samples from each term
    sampled_dfs = []
    remaining_samples = total_samples - (min_samples_per_term * num_terms)
    
    if remaining_samples < 0:
        logger.warning(f"Total samples ({total_samples}) is less than minimum required ({min_samples_per_term * num_terms}). Adjusting min samples.")
        min_samples_per_term = total_samples // num_terms
        remaining_samples = total_samples - (min_samples_per_term * num_terms)
    
    # Calculate term weights based on average density ratios
    term_weights = {}
    for term in all_terms:
        # Use the average density ratio for the term as its weight
        term_weights[term] = np.mean(term_density_ratios_dict[term])
    
    # Normalize weights to sum to 1
    total_weight = sum(term_weights.values())
    for term in term_weights:
        term_weights[term] /= total_weight
    
    # Log the term weights
    logger.info("Term weights for additional sampling:")
    for term, weight in term_weights.items():
        logger.info(f"  {term}: {weight:.4f}")
    
    # Track sample counts to enforce max_repetitions
    sample_counts = {}
    
    # For each term, sample based on density ratios
    for term in all_terms:
        df = term_dataset_dict[term]
        density_ratios = term_density_ratios_dict[term]
        
        # Ensure there are enough samples
        if len(df) < min_samples_per_term / max_repetitions:
            logger.warning(f"Term '{term}' has too few unique samples ({len(df)}) to meet minimum required ({min_samples_per_term}) without exceeding max repetitions. Using all available samples with max repetitions.")
            # Use all samples repeated max_repetitions times
            repeated_df = pd.concat([df] * max_repetitions)
            repeated_df = repeated_df.head(min_samples_per_term)
            sampled_dfs.append(repeated_df)
            
            # Track these samples
            for _, row in df.iterrows():
                sample_id = (term, row.name)
                sample_counts[sample_id] = sample_counts.get(sample_id, 0) + min(max_repetitions, min_samples_per_term // len(df) + 1)
            continue
        
        # First, sample the minimum number from this term using controlled sampling
        df_copy = df.copy()
        df_copy['sampling_weight'] = np.exp(density_ratios - np.max(density_ratios))
        df_copy['log_density_ratio'] = density_ratios
        
        # Sample with controlled replacement
        min_sampled = sample_with_max_repetitions(
            df_copy, 
            min_samples_per_term, 
            'sampling_weight', 
            max_repetitions,
            sample_counts,
            term
        )
        sampled_dfs.append(min_sampled)
    
    # Now distribute the remaining samples based on term weights
    for term in all_terms:
        additional_samples = int(remaining_samples * term_weights[term])
        if additional_samples <= 0:
            continue
            
        df = term_dataset_dict[term]
        density_ratios = term_density_ratios_dict[term]
        
        # Prepare for sampling
        df_copy = df.copy()
        df_copy['sampling_weight'] = np.exp(density_ratios - np.max(density_ratios))
        df_copy['log_density_ratio'] = density_ratios
        
        # Sample additional samples with controlled replacement
        additional_sampled = sample_with_max_repetitions(
            df_copy, 
            additional_samples, 
            'sampling_weight', 
            max_repetitions,
            sample_counts,
            term
        )
        sampled_dfs.append(additional_sampled)
    
    # Combine all sampled dataframes
    sampled_df = pd.concat(sampled_dfs, ignore_index=True)
    
    # Shuffle the resulting dataset
    sampled_df = sampled_df.sample(frac=1).reset_index(drop=True)
    
    # Calculate and log repetition statistics
    repetition_counts = {}
    for sample_id, count in sample_counts.items():
        term = sample_id[0]
        repetition_counts[term] = repetition_counts.get(term, {})
        repetition_counts[term][count] = repetition_counts[term].get(count, 0) + 1
    
    logger.info("Sample repetition statistics by term:")
    for term, counts in repetition_counts.items():
        logger.info(f"  {term}:")
        for rep_count, num_samples in sorted(counts.items()):
            logger.info(f"    {num_samples} samples repeated {rep_count} times")
    
    return sampled_df

def sample_with_max_repetitions(df, n_samples, weight_col, max_repetitions, sample_counts, term):
    """
    Sample from a dataframe with a maximum number of repetitions per sample.
    
    Parameters:
    -----------
    df: dataframe to sample from
    n_samples: number of samples to draw
    weight_col: column name containing weights for weighted sampling
    max_repetitions: maximum number of times a single sample can be repeated
    sample_counts: dictionary tracking how many times each sample has been selected
    term: the current term being sampled
    
    Returns:
    --------
    sampled_df: dataframe with samples
    """
    result = []
    remaining = n_samples
    
    # Keep track of indices that have reached max_repetitions
    excluded_indices = set()
    
    while remaining > 0 and len(excluded_indices) < len(df):
        # Create a mask for eligible samples
        eligible = ~df.index.isin(excluded_indices)
        
        if not any(eligible):
            logger.warning(f"No more eligible samples for term {term}. Some samples may not reach target count.")
            break
        
        # Get eligible samples
        eligible_df = df[eligible]
        eligible_weights = eligible_df[weight_col]
        
        # Sample one at a time to carefully control repetition
        batch_size = min(remaining, sum(eligible))
        
        try:
            # Sample without replacement within this batch
            batch = eligible_df.sample(
                n=batch_size,
                weights=eligible_weights,
                replace=False
            )
            
            # Check each sample against max_repetitions
            for idx, row in batch.iterrows():
                sample_id = (term, idx)
                current_count = sample_counts.get(sample_id, 0)
                
                if current_count < max_repetitions:
                    result.append(row)
                    remaining -= 1
                    sample_counts[sample_id] = current_count + 1
                    
                    # If this sample reaches max_repetitions, exclude it from future sampling
                    if current_count + 1 >= max_repetitions:
                        excluded_indices.add(idx)
                else:
                    # This sample has already reached max_repetitions
                    excluded_indices.add(idx)
        except ValueError as e:
            logger.warning(f"Sampling error: {e}. Adjusting approach.")
            # If weighted sampling fails, try uniform sampling as fallback
            available = eligible_df.index.tolist()
            if available:
                selected = np.random.choice(available)
                batch = eligible_df.loc[[selected]]
                
                for idx, row in batch.iterrows():
                    sample_id = (term, idx)
                    current_count = sample_counts.get(sample_id, 0)
                    
                    if current_count < max_repetitions:
                        result.append(row)
                        remaining -= 1
                        sample_counts[sample_id] = current_count + 1
                        
                        if current_count + 1 >= max_repetitions:
                            excluded_indices.add(idx)
            else:
                break
    
    # Convert result list to DataFrame
    if result:
        return pd.DataFrame(result)
    else:
        return pd.DataFrame(columns=df.columns)

def visualize_term_distribution(sampled_df):
    """
    Create visualizations of the term distribution in the sampled dataset.
    
    Parameters:
    -----------
    sampled_df: the sampled dataset
    """
    # Create output directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Plot term distribution
    plt.figure(figsize=(14, 8))
    term_counts = sampled_df['term'].value_counts()
    sns.barplot(x=term_counts.index, y=term_counts.values)
    plt.title('Term Distribution in Sampled Dataset')
    plt.xlabel('Term')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('plots/term_distribution.png')
    
    # Plot label distribution within each term
    plt.figure(figsize=(14, 8))
    term_label_counts = sampled_df.groupby(['term', 'label']).size().unstack(fill_value=0)
    term_label_counts.plot(kind='bar', stacked=True, figsize=(14, 8))
    plt.title('Label Distribution by Term in Sampled Dataset')
    plt.xlabel('Term')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Label')
    plt.tight_layout()
    plt.savefig('plots/label_distribution_by_term.png')
    
    # Plot density ratio distribution
    plt.figure(figsize=(14, 8))
    sns.histplot(sampled_df['log_density_ratio'], bins=50, kde=True)
    plt.title('Log Density Ratio Distribution in Sampled Dataset')
    plt.xlabel('Log Density Ratio')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('plots/density_ratio_distribution.png')
    
    logger.info("Saved visualizations to 'plots' directory.")

def main():
    """
    Main function to run the density ratio sampling with term balancing.
    """
    # Hardcoded parameters
    k = 1000  # Number of neighbors for KNN density estimation
    total_samples = 50000  # Total number of samples in the output dataset
    min_samples_per_term = 500  # Minimum number of samples per term
    max_repetitions = 50  # Maximum number of times a sample can be repeated
    output_path = 'new_data/balanced_term_samples.csv'  # Output path
    
    logger.info("Starting density ratio sampling with term balancing...")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load datasets and embeddings
    russian_embeddings, term_embeddings_dict, term_dataset_dict = load_datasets_and_embeddings()
    
    # Calculate density ratios for each term
    term_density_ratios_dict = calculate_term_density_ratios(
        term_embeddings_dict,
        term_dataset_dict,
        russian_embeddings,
        k=k
    )
      # Create balanced sampled dataset
    logger.info(f"Creating balanced sampled dataset with {total_samples} total samples...")
    sampled_df = create_balanced_sampled_dataset(
        term_density_ratios_dict,
        term_dataset_dict,
        total_samples=total_samples,
        min_samples_per_term=min_samples_per_term,
        max_repetitions=max_repetitions
    )
    
    # Save the results
    logger.info(f"Saving results to {output_path}")
    sampled_df.to_csv(output_path, index=False)
    
    # Display term distribution statistics
    logger.info("\nTerm Distribution in Sampled Dataset:")
    term_distribution = sampled_df['term'].value_counts()
    for term, count in term_distribution.items():
        logger.info(f"  {term}: {count} samples ({count/len(sampled_df)*100:.1f}%)")
    
    # Display label distribution statistics
    logger.info("\nLabel Distribution in Sampled Dataset:")
    label_distribution = sampled_df['label'].value_counts()
    for label, count in label_distribution.items():
        logger.info(f"  {label}: {count} samples ({count/len(sampled_df)*100:.1f}%)")
    
    # Display log density ratio statistics
    logger.info("\nLog Density Ratio Statistics:")
    logger.info(f"  Min: {sampled_df['log_density_ratio'].min():.6f}")
    logger.info(f"  Max: {sampled_df['log_density_ratio'].max():.6f}")
    logger.info(f"  Mean: {sampled_df['log_density_ratio'].mean():.6f}")
    logger.info(f"  Median: {sampled_df['log_density_ratio'].median():.6f}")
    
    # Create visualizations
    visualize_term_distribution(sampled_df)
    
    logger.info("\nSuccess! The process has completed.")
    logger.info(f"Balanced term dataset has been saved to: {output_path}")

if __name__ == "__main__":
    main()
