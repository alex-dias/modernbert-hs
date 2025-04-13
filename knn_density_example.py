"""
This script demonstrates how to use the KNN Density Estimator to select the most similar
subset of dataset B to dataset A.

It loads example data from the toxigen datasets and runs the KNN density estimation.
"""

import sys
from knn_density_estimator import run_knn_density_estimation

def main():
    """Run a simple example of KNN density estimation."""
    print("KNN Density Estimator Example")
    print("=============================")
    
    # Create dummy example embeddings if they don't exist
    # Note: In a real scenario, you would use actual embeddings
    embA_path = 'embeddings/embeddings_set_A.npy'
    embB_path = 'embeddings/embeddings_set_B.npy'
    
    # Run the KNN density estimator
    print("\nRunning KNN Density Estimator...")
    k = 5  
    percent = 0.1 
    output_path = 'selected_samples.csv'
    
    # Call the function directly instead of using os.system
    print(f"Running KNN density estimation with k={k}, percent={percent}")
    
    # Parameters for the function call
    dataset_B_path = 'embeddings/dataset_B.csv'
    
    # Call the function
    results = run_knn_density_estimation(
        embeddings_a_path=embA_path,
        embeddings_b_path=embB_path,
        text_b_path=dataset_B_path,
        k=k,
        percent=percent,
        output_path=f"selected_knn/samples_knn_k{k}_percent{int(percent*100)}.csv",
        use_gpu=True
    )
    
    if results is None:
        print("Error: KNN density estimation failed.")
        sys.exit(1)
    
    # Display results
    print(f"\nSelected {len(results)} samples from dataset B")
    print("\nSample of selected data:")
    print(results[['text', 'term', 'density']].head())
    
    print("\nDistribution of selected samples by term:")
    term_counts = results['term'].value_counts()
    for term, count in term_counts.items():
        print(f"  {term}: {count} samples")
    
    print("\nSuccess! The example has completed.")
    print(f"Selected samples have been saved to: {output_path}")

if __name__ == "__main__":
    main()