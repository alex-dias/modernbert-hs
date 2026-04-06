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

# ... [load_term_embeddings and calculate_density functions remain unchanged] ...

def load_term_embeddings(term, base_path='term_embeddings_v2/sentence-transformers_all-mpnet-base-v2'):
    """Load embeddings for a specific term."""
    embedding_path = os.path.join(base_path, f'{term}.npy')
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
        return np.mean(density_log)
    except Exception as e:
        logger.error(f"Error calculating density: {e}")
        return None

def plot_density_heatmap(density_matrix, terms, output_path="images/all_term_densities_heatmap"):
    """
    Generate a heatmap visualization of the density matrix using IEEE standards.
    """
    # --- IEEE PLOT CONFIGURATION ---
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9
    plt.rcParams['figure.titlesize'] = 12
    
    # IEEE Double Column width is ~7.16 inches
    plt.figure(figsize=(7.16, 6))
    
    # Create a mask for the diagonal (self-comparison)
    mask = np.zeros_like(density_matrix, dtype=bool)
    np.fill_diagonal(mask, True)
    
    # Create heatmap
    # Using 'viridis' to match the previous IEEE style, but 'YlGnBu' is also acceptable.
    ax = sns.heatmap(
        density_matrix,
        xticklabels=terms,
        yticklabels=terms,
        annot=True,
        fmt=".1f",      # Increased precision to 2 decimals for scientific accuracy
        mask=mask,      # Mask the diagonal
        cmap="viridis", # Perceptually uniform colormap
        linewidths=0.5, # Distinct cell borders
        annot_kws={"size": 7}, # Small text to fit in cells
        cbar_kws={'label': 'Average Density'}
    )
    
    # Set labels (Bold is optional but common for axis titles)
    plt.xlabel('Target Term Dataset', fontweight='bold')
    plt.ylabel('Source Term Dataset', fontweight='bold')
    
    # Note: Title removed as per IEEE standards (use LaTeX figure caption instead)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure in both PDF (Vector) and PNG (Raster)
    # Removing extension from output_path if present to save both formats
    base_path = os.path.splitext(output_path)[0]
    
    plt.savefig(f"{base_path}.pdf", format='pdf', bbox_inches='tight')
    plt.savefig(f"{base_path}.png", dpi=300, bbox_inches='tight')
    
    logger.info(f"Density heatmap saved to {base_path}.png and {base_path}.pdf")
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
            
            avg_density = calculate_density(source_embeddings, target_embeddings)
            if avg_density is not None:
                density_matrix[i, j] = avg_density
            else:
                density_matrix[i, j] = 0
    
    # Create DataFrame
    density_df = pd.DataFrame(density_matrix, index=available_terms, columns=available_terms)
    density_df.to_csv('all_term_densities.csv')
    logger.info("Density matrix saved to all_term_densities.csv")
    
    # Plot heatmap
    display_terms = [term if term != 'russian' else 'slavic' for term in available_terms]
    plot_density_heatmap(density_matrix, display_terms)
    
    # Print highest density pairs
    logger.info("\nHighest density pairs:")
    non_diag_indices = ~np.eye(n_terms, dtype=bool)
    flattened_density = density_matrix[non_diag_indices]
    flattened_indices = np.argwhere(non_diag_indices)
    
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