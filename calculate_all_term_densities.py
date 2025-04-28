"""
Calculate KNN density between all term datasets and visualize the results as a heatmap.

This script computes the KNN density between each pair of term datasets, creating a matrix
of density values that represents how similar each term dataset is to every other term dataset.
The results are visualized as a heatmap for easy interpretation.
"""

import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from dataHandler import getListOfIdTerms
from knn_density_estimator import faiss_knn_density
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('All-Term-Density-Calculator')

def load_term_embeddings(term, base_path='term_embeddings'):
    """Load embeddings for a specific term."""
    embedding_path = os.path.join(base_path, f'embeddings_{term}.npy')
    if not os.path.exists(embedding_path):
        logger.warning(f"No embeddings found for term '{term}' at {embedding_path}")
        return None
    
    try:
        embeddings = np.load(embedding_path)
        logger.info(f"Loaded {term} embeddings: {embeddings.shape}")
        return embeddings
    except Exception as e:
        logger.error(f"Error loading embeddings for {term}: {e}")
        return None

def calculate_density(source_embeddings, target_embeddings, k=1000, use_gpu=True):
    """Calculate KNN density from source embeddings to target embeddings."""
    try:
        _, _, density_log = faiss_knn_density(
            source_embeddings, 
            target_embeddings, 
            k, 
            normalize_vectors=True, 
            use_gpu=use_gpu
        )
        
        # Return the mean density
        return np.mean(density_log)
    except Exception as e:
        logger.error(f"Error calculating density: {e}")
        return None

def plot_density_heatmap(density_matrix, terms, output_path="all_term_densities_heatmap.png"):
    """Generate a heatmap visualization of the density matrix."""
    plt.figure(figsize=(12, 10))
    
    # Create a mask for the diagonal (self-comparison)
    mask = np.zeros_like(density_matrix, dtype=bool)
    np.fill_diagonal(mask, True)
    
    # Create heatmap with custom colormap
    ax = sns.heatmap(
        density_matrix,
        xticklabels=terms,
        yticklabels=terms,
        annot=True,
        fmt=".1f",  
        mask=mask,  # Mask the diagonal
        cmap="YlGnBu",
        cbar_kws={'label': 'Average Density'}
    )
    
    # Set labels and title
    plt.xlabel('Target Term Dataset')
    plt.ylabel('Source Term Dataset')
    plt.title('Average KNN Density Between Term Datasets')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Density heatmap saved to {output_path}")
    
    plt.close()

def main():
    """Main function to calculate densities between all term datasets."""
    logger.info("Starting density calculation between all term datasets...")
    
    # Get all terms
    terms = getListOfIdTerms()
    logger.info(f"Will process {len(terms)} terms: {', '.join(terms)}")
    
    # Load all embeddings
    embeddings_dict = {}
    for term in terms:
        emb = load_term_embeddings(term)
        if emb is not None:
            embeddings_dict[term] = emb
    
    available_terms = list(embeddings_dict.keys())
    logger.info(f"Successfully loaded embeddings for {len(available_terms)} terms: {', '.join(available_terms)}")
    
    if not available_terms:
        logger.error("No embeddings were loaded. Exiting.")
        return
    
    # Initialize density matrix
    n_terms = len(available_terms)
    density_matrix = np.zeros((n_terms, n_terms))
    
    # Calculate density for each pair of terms
    logger.info("Calculating density for all term pairs...")
    for i, source_term in enumerate(tqdm(available_terms)):
        source_embeddings = embeddings_dict[source_term]
        
        for j, target_term in enumerate(available_terms):
            target_embeddings = embeddings_dict[target_term]
            
            # Skip self-comparison (diagonal)
            if source_term == target_term:
                density_matrix[i, j] = 0
                continue
            
            # Calculate average density from source to target
            avg_density = calculate_density(source_embeddings, target_embeddings)
            
            if avg_density is not None:
                density_matrix[i, j] = avg_density
            else:
                logger.warning(f"Could not calculate density from {source_term} to {target_term}")
                density_matrix[i, j] = 0
    
    # Create a DataFrame for the density matrix
    density_df = pd.DataFrame(density_matrix, index=available_terms, columns=available_terms)
    
    # Save the density matrix to CSV
    density_df.to_csv('all_term_densities.csv')
    logger.info("Density matrix saved to all_term_densities.csv")
    
    # Plot heatmap
    plot_density_heatmap(density_matrix, available_terms)
    
    # Find and print the terms with highest average density to each other
    logger.info("\nHighest density pairs:")
    
    # Create a flattened view of the non-diagonal elements
    non_diag_indices = ~np.eye(n_terms, dtype=bool)
    flattened_density = density_matrix[non_diag_indices]
    flattened_indices = np.argwhere(non_diag_indices)
    
    # Get the top 5 highest density pairs
    top_n = min(5, len(flattened_density))
    top_indices = np.argsort(flattened_density)[-top_n:][::-1]
    
    for idx in top_indices:
        i, j = flattened_indices[idx]
        source = available_terms[i]
        target = available_terms[j]
        density = density_matrix[i, j]
        logger.info(f"  {source} → {target}: {density:.6f}")
    
    logger.info("\nComplete!")
    return density_df

if __name__ == "__main__":
    density_df = main()