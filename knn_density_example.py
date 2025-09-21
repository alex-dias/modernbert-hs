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
    k = [5, 100, 1000]
    bal = [True, False]
    percent = 1
    
    for k_val in k:
        for balanced in bal:
            print(f"\nRunning KNN density estimation with k={k_val}, balanced={balanced}")
            
            # Parameters for the function call
            dataset_B_path = 'embeddings/dataset_B.csv'
            
            if not balanced:
                output_path=f"selected_knn_new/samples_knn_k{k_val}_percent{int(percent*100)}.csv"
            else:
                output_path=f"selected_knn_new/samples_knn_k{k_val}_percent{int(percent*100)}_balanced.csv"
            
            # Call the function
            results = run_knn_density_estimation(
                embeddings_a_path=embA_path,
                embeddings_b_path=embB_path,
                text_b_path=dataset_B_path,
                k=k_val,
                percent=percent,
                output_path=output_path,
                use_gpu=True,
                balanced=balanced
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