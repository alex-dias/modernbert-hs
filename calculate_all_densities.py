import pandas as pd
import numpy as np
import os
import logging
from tqdm import tqdm
from dataHandler import getListOfIdTerms, getToxigenDatasetListClass
from knn_density_estimator import faiss_knn_density

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('All-Densities-Calculator')

def load_embeddings(term, base_path='term_embeddings_v2/sentence-transformers_all-mpnet-base-v2'):
    """Load embeddings for a specific term."""
    embedding_path = os.path.join(base_path, f'{term}.npy')
    try:
        if os.path.exists(embedding_path):
            return np.load(embedding_path)
        else:
            logger.error(f"Embeddings not found for {term} at {embedding_path}")
            return None
    except Exception as e:
        logger.error(f"Error loading embeddings for {term}: {e}")
        return None

def main():
    # 1. Setup
    OUTPUT_DIR = 'select_knn_complete_2'
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    K_VALUES = [5, 100, 1000]
    
    # 2. Reconstruct the "Original Dataset"
    logger.info("Reconstructing original dataset...")
    # This gets [text, term, label] for all terms except russian
    df_dataset = getToxigenDatasetListClass(is_random=False) 
    
    # List of terms in the dataset (order matters for concatenating embeddings)
    # getListOfIdTerms() returns ['asian', ... 'russian']
    # getToxigenDatasetListClass uses getListOfIdTerms()[:-1]
    source_terms = getListOfIdTerms()[:-1] 
    
    logger.info(f"Dataset reconstructed. Shape: {df_dataset.shape}")
    logger.info(f"Source terms: {source_terms}")

    # 3. Reconstruct Source Embeddings
    logger.info("Loading source embeddings...")
    source_embeddings_list = []
    
    for term in source_terms:
        emb = load_embeddings(term)
        if emb is None:
            logger.error(f"Critical error: Missing embeddings for source term {term}")
            return
        source_embeddings_list.append(emb)
        
    source_embeddings = np.concatenate(source_embeddings_list, axis=0)
    logger.info(f"Source embeddings concatenated. Shape: {source_embeddings.shape}")
    
    if len(source_embeddings) != len(df_dataset):
        logger.error(f"Mismatch! Dataset has {len(df_dataset)} rows but embeddings have {len(source_embeddings)} rows.")
        # This usually happens if the CSVs used in dataHandler differ from what generated the embeddings.
        # Assuming they match for now based on previous files.
        return

    # 4. Define Target Terms
    # The user wants: asian, black, chinese, jewish, lation (latino), lgbtq, mental_dis, mexican, 
    # middle_east, muslim, native_american, physical_dis, russian, women.
    # This is exactly getListOfIdTerms().
    target_terms = getListOfIdTerms()
    
    # Pre-load all target embeddings to avoid reloading in loops
    target_embeddings_map = {}
    logger.info("Loading target embeddings...")
    for term in target_terms:
        # User wrote 'lation' in request, but file is 'latino'. Mapping 'lation' -> 'latino' logic implies just using 'latino'.
        # The list from dataHandler has 'latino'.
        emb = load_embeddings(term)
        if emb is not None:
            target_embeddings_map[term] = emb
            
    # Also we need 'russian' which is the last one in getListOfIdTerms()
    # Check if we successfully loaded everything
    logger.info(f"Loaded {len(target_embeddings_map)} target embedding sets.")

    # 5. Process for each K
    for k in K_VALUES:
        logger.info(f"Processing for K={k}...")
        
        # Create a copy of the dataframe for this K
        df_k = df_dataset.copy()
        
        # Calculate density for each target term
        for term in target_terms:
            logger.info(f"  Calculating density for target: {term}")
            if term not in target_embeddings_map:
                logger.warning(f"  Skipping {term} (no embeddings)")
                df_k[f'density_{term}'] = np.nan
                continue
                
            target_emb = target_embeddings_map[term]
            
            # density of source_embeddings relative to target_emb
            # faiss_knn_density returns distances, indices, density_log
            _, _, density_log = faiss_knn_density(
                source_embeddings, 
                target_emb, 
                k, 
                normalize_vectors=True, 
                use_gpu=True # Assuming GPU available as in other scripts, otherwise set False
            )
            
            df_k[f'density_{term}'] = density_log
            
        # Calculate density against ALL samples (source_embeddings itself)
        logger.info(f"  Calculating density for target: ALL samples")
        _, _, density_all = faiss_knn_density(
            source_embeddings, 
            source_embeddings, # Reference is itself
            k, 
            normalize_vectors=True, 
            use_gpu=True
        )
        df_k['density_all'] = density_all
        
        # Rename 'latino' to 'lation' if strictly following the request text? 
        # User request: "asian... lation... mexican..."
        # But user also said "similar to samples_knn_k5_percent100.csv".
        # Let's check the request: "columns text, label, term and the density to each one of the other terms: asian, ... lation ..."
        # I will rename the column `density_latino` to `density_lation` to match request perfectly if needed, 
        # but standardizing on `latino` (the real filename) is safer for code, but user asked for `lation`.
        # I'll keep `density_latino` as it matches the source term name. If user *really* wants lation I can change it.
        # Given "lation, lgbtq..." looks like a typo in user prompt, I will stick to the correct "latino".
        
        # Save
        output_filename = f"samplex_knnratio_k{k}.csv"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        df_k.to_csv(output_path, index=False)
        logger.info(f"  Saved {output_path}")

    logger.info("All finished.")

if __name__ == "__main__":
    main()
