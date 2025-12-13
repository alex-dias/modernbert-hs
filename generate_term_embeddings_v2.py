"""
Generate embeddings for each term dataset using dataHandler.py.

This script loads the text data for each term using dataHandler.toxigenDataset,
generates embeddings using a pre-trained model from Sentence Transformers,
and saves the embeddings to the term_embeddings_v2 folder.
"""

import os
import numpy as np
import logging
import sys
from sentence_transformers import SentenceTransformer
from dataHandler import getListOfIdTerms, toxigenDataset, getCompleteRussDataset

# ==========================================
# CONFIGURATION
# ==========================================
# Model to use for generating embeddings
# Examples: 'all-MiniLM-L6-v2', 'bert-base-uncased', 'roberta-base'
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2" 

# Batch size for processing texts
BATCH_SIZE = 32
# ==========================================

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('Embedding-Generator-V2')

def get_sanitized_model_name(model_name):
    """Sanitize model name for use as a folder name."""
    return model_name.replace("/", "_").replace("\\", "_")

def generate_and_save_embeddings():
    """Main function to generate embeddings for all terms."""
    logger.info("Starting embedding generation (V2)...")
    logger.info(f"Target Model: {MODEL_NAME}")
    
    # 1. Prepare Output Directory
    sanitized_model_name = get_sanitized_model_name(MODEL_NAME)
    output_dir = os.path.join('term_embeddings_v2', sanitized_model_name)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # 2. Load Model
    try:
        logger.info(f"Loading model: {MODEL_NAME}")
        model = SentenceTransformer(MODEL_NAME)
    except Exception as e:
        logger.error(f"Error loading model {MODEL_NAME}: {e}")
        sys.exit(1)

    # 3. Get Terms
    terms = getListOfIdTerms()
    # Filter out 'russian' if it's not in the toxigen dataset structure usually
    # dataHandler.getListOfIdTerms returns 'russian' at the end, let's see if toxigenDataset handles it.
    # Looking at dataHandler.py types, toxigenDataset reads {term}.csv. 
    # If russian.csv exists in new_data, it will work. 
    # Previous code skipped the last one in one function, but let's try all and handle errors.
    
    logger.info(f"Found {len(terms)} terms to process: {', '.join(terms)}")

    # 4. Process Each Term
    for term in terms:
        logger.info(f"Processing term: {term}")
        
        try:
            # Load Dataset
            if term == 'russian':
                logger.info("Using getCompleteRussDataset for 'russian' term")
                all_texts = getCompleteRussDataset()
            else:
                # toxigenDataset returns a DatasetDict with 'train' and 'test'
                # We want all data, so we combine them.
                dataset_dict = toxigenDataset(term)
                texts_train = dataset_dict['train']['text']
                texts_test = dataset_dict['test']['text']
                all_texts = texts_train + texts_test
            
            if not all_texts:
                logger.warning(f"No texts found for term '{term}'")
                continue
                
            logger.info(f"Loaded {len(all_texts)} texts for term: {term}")

            # Generate Embeddings
            embeddings = model.encode(all_texts, show_progress_bar=True, batch_size=BATCH_SIZE)
            logger.info(f"Generated embeddings shape: {embeddings.shape}")

            # Save Embeddings
            output_file = os.path.join(output_dir, f'{term}.npy')
            np.save(output_file, embeddings)
            logger.info(f"Saved to: {output_file}")
            
        except FileNotFoundError:
             logger.warning(f"Data file for term '{term}' not found. Skipping.")
        except Exception as e:
            logger.error(f"Error processing term '{term}': {e}")
            continue

    logger.info("Embedding generation completed successfully!")

if __name__ == "__main__":
    generate_and_save_embeddings()
