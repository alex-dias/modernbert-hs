"""
Calculate average and median K-nearest neighbor density of each dataset term relative to the Russian dataset.

This script computes the average and median density values between embeddings of different terms
and the Russian dataset, providing insights into which terms have the highest density in relation
to the Russian dataset in the embedding space.
"""

import numpy as np
import pandas as pd
import os
import sys
from dataHandler import getListOfIdTerms
from knn_density_estimator import faiss_knn_density
import matplotlib.pyplot as plt
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('Term-Density-Calculator')

def load_term_embeddings(term, base_path='term_embeddings'):
    """Load embeddings for a specific term."""
    embedding_path = os.path.join(base_path, f'embeddings_{term}.npy')
    if not os.path.exists(embedding_path):
        logger.warning(f"No embeddings found for term '{term}' at {embedding_path}")
        return None
    
    try:
        return np.load(embedding_path)
    except Exception as e:
        logger.error(f"Error loading embeddings for {term}: {e}")
        return None

def calculate_densities_to_russian(term_embeddings, russian_embeddings, k=100, use_gpu=True):
    """Calculate k-nn densities from term embeddings to Russian embeddings."""
    try:
        _, _, density_log = faiss_knn_density(
            term_embeddings, 
            russian_embeddings, 
            k, 
            normalize_vectors=True, 
            use_gpu=use_gpu
        )
        
        return density_log
    except Exception as e:
        logger.error(f"Error calculating densities: {e}")
        return None

def calculate_statistics(densities):
    """Calculate average and median densities."""
    if densities is None:
        return None, None
    
    # Calculate average and median of densities across all points
    avg_density = np.mean(densities)
    median_density = np.median(densities)
    
    return avg_density, median_density

def plot_density_statistics(term_stats, output_path="term_densities.png"):
    """Generate a bar plot of average and median densities for each term."""
    terms = list(term_stats.keys())
    avgs = [stats['average'] for stats in term_stats.values()]
    medians = [stats['median'] for stats in term_stats.values()]
    
    plt.figure(figsize=(12, 8))
    
    x = np.arange(len(terms))
    width = 0.35
    
    plt.bar(x - width/2, avgs, width, label='Average Density')
    plt.bar(x + width/2, medians, width, label='Median Density')
    
    plt.xlabel('Dataset Terms')
    plt.ylabel('Density relative to Russian Dataset')
    plt.title('Average and Median Densities relative to Russian Dataset')
    plt.xticks(x, terms, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(output_path)
    logger.info(f"Density statistics plot saved to {output_path}")
    plt.close()

def main():
    """Main function to run the density calculations."""
    logger.info("Starting density calculation...")
    
    # Load Russian embeddings
    russian_emb_path = 'term_embeddings/embeddings_russian.npy'
    if not os.path.exists(russian_emb_path):
        logger.error(f"Russian embeddings not found at {russian_emb_path}")
        sys.exit(1)
        
    try:
        russian_embeddings = np.load(russian_emb_path)
        logger.info(f"Loaded Russian embeddings: {russian_embeddings.shape}")
    except Exception as e:
        logger.error(f"Error loading Russian embeddings: {e}")
        sys.exit(1)
    
    # Get all terms except Russian
    terms = [term for term in getListOfIdTerms() if term != 'russian']
    logger.info(f"Will process {len(terms)} terms: {', '.join(terms)}")
    
    # Dictionary to store results
    term_stats = {}
    
    # Process each term
    for term in terms:
        logger.info(f"Processing term: {term}")
        
        # Load term embeddings
        term_embeddings = load_term_embeddings(term)
        if term_embeddings is None:
            logger.warning(f"Skipping term '{term}' due to missing embeddings")
            continue
            
        logger.info(f"Loaded {term} embeddings: {term_embeddings.shape}")
        
        # Calculate densities
        densities = calculate_densities_to_russian(term_embeddings, russian_embeddings)
        if densities is None:
            logger.warning(f"Skipping term '{term}' due to error in density calculation")
            continue
        
        # Calculate statistics
        avg_density, median_density = calculate_statistics(densities)
        
        # Store results
        term_stats[term] = {
            'average': float(avg_density),
            'median': float(median_density)
        }
        
        logger.info(f"Term: {term}, Avg Density: {avg_density:.6f}, Median Density: {median_density:.6f}")
    
    # Sort terms by average density (descending, since higher density means more similar)
    sorted_terms = {k: v for k, v in sorted(term_stats.items(), key=lambda item: item[1]['average'], reverse=True)}
    
    # Print sorted results
    logger.info("\nResults sorted by average density (most similar to least similar to Russian dataset):")
    for term, stats in sorted_terms.items():
        logger.info(f"{term}: Avg={stats['average']:.6f}, Median={stats['median']:.6f}")
    
    # Generate plot
    plot_density_statistics(term_stats)
    
    # Save results to CSV
    results_df = pd.DataFrame.from_dict(term_stats, orient='index').reset_index()
    results_df.rename(columns={'index': 'term'}, inplace=True)
    results_df.to_csv('term_densities_to_russian.csv', index=False)
    logger.info("Results saved to term_densities_to_russian.csv")
    
    return term_stats

if __name__ == "__main__":
    term_stats = main()