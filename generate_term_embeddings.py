"""
Generate embeddings for each term dataset from the masked_data folder.

This script loads the text data for each term from the masked_data folder,
generates embeddings using a pre-trained model from Sentence Transformers,
and saves the embeddings to the term_embeddings folder.
"""

import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from dataHandler import getListOfIdTerms
import logging
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('Embedding-Generator')

def load_masked_data(term, base_path='masked_data'):
    """Load masked data for a specific term."""
    file_path = os.path.join(base_path, f'toxigen_masked_pred_{term}.csv')
    if not os.path.exists(file_path):
        logger.warning(f"No masked data found for term '{term}' at {file_path}")
        return None
    
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} rows for term: {term}")
        # Check if the dataframe has a text column
        if "text" not in df.columns:
            # Try to find the column containing the text data
            text_column = None
            potential_columns = ["text", "Tweet", "tweet", "content", "Text"]
            for col in potential_columns:
                if col in df.columns:
                    text_column = col
                    break
            
            if text_column is None:
                logger.warning(f"Could not find text column in dataframe for {term}. Available columns: {df.columns.tolist()}")
                # Take the first column that's not 'label' or 'term'
                for col in df.columns:
                    if col.lower() not in ['label', 'term']:
                        text_column = col
                        break
            
            logger.info(f"Using column '{text_column}' for text content for term {term}")
            return df[text_column].tolist()
        else:
            return df["text"].tolist()
            
    except Exception as e:
        logger.error(f"Error loading masked data for {term}: {e}")
        return None

def generate_embeddings(texts, model_name="all-MiniLM-L6-v2"):
    """Generate embeddings for a list of texts using a pre-trained model."""
    if not texts:
        logger.warning("No texts provided for embedding generation")
        return None
    
    try:
        logger.info(f"Loading model: {model_name}")
        model = SentenceTransformer(model_name)
        
        logger.info(f"Generating embeddings for {len(texts)} texts")
        embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
        
        logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        return None

def save_embeddings(embeddings, term, output_dir='term_embeddings'):
    """Save embeddings to a file."""
    if embeddings is None:
        logger.warning(f"No embeddings to save for term: {term}")
        return False
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f'embeddings_{term}.npy')
    try:
        np.save(output_path, embeddings)
        logger.info(f"Saved embeddings for {term} to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving embeddings for {term}: {e}")
        return False

def main():
    """Main function to generate embeddings for all terms."""
    logger.info("Starting embedding generation...")
    
    # Ensure output directory exists
    os.makedirs('term_embeddings', exist_ok=True)
    
    # Get list of all terms
    terms = getListOfIdTerms()
    logger.info(f"Will process {len(terms)} terms: {', '.join(terms)}")
    
    # Initialize model once to avoid reloading for each term
    model_name = "all-MiniLM-L6-v2"  # Good balance of speed and quality
    try:
        logger.info(f"Loading model: {model_name}")
        model = SentenceTransformer(model_name)
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        sys.exit(1)
    
    # Process each term
    for term in terms:
        logger.info(f"Processing term: {term}")
        
        # Load masked data
        texts = load_masked_data(term)
        if texts is None or len(texts) == 0:
            logger.warning(f"Skipping term '{term}' due to missing data")
            continue
        
        logger.info(f"Generating embeddings for {len(texts)} texts from term: {term}")
        
        # Generate embeddings
        try:
            embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
            logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        except Exception as e:
            logger.error(f"Error generating embeddings for {term}: {e}")
            continue
        
        # Save embeddings
        save_embeddings(embeddings, term)
    
    logger.info("Embedding generation completed successfully!")

if __name__ == "__main__":
    main()