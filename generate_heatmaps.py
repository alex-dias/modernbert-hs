import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_evaluation_results(json_path):
    """
    Load evaluation results from JSON file.
    
    Parameters:
    - json_path: Path to the JSON file containing cross-term evaluation results
    
    Returns:
    - Dictionary of evaluation results
    """
    with open(json_path, "r") as f:
        results = json.load(f)
    return results

def generate_heatmaps(results, output_dir="cross_term_heatmaps", selected_metrics=None):
    """
    Generate heatmap visualizations from evaluation results.
    
    Parameters:
    - results: Dictionary of evaluation results
    - output_dir: Directory to save output files
    - selected_metrics: List of specific metrics to visualize (defaults to all available metrics)
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Get list of models and terms
    terms = list(results.keys())
    print(f"Found {len(terms)} terms/models in results")
    
    # Determine which metrics are available
    metrics = set()
    for model_term in terms:
        for dataset_term in results[model_term]:
            metrics.update(results[model_term][dataset_term].keys())
    
    metrics = [m for m in metrics if m not in ["true_positives", "false_positives", "true_negatives", "false_negatives"]]
    metrics = sorted(list(metrics))
    print(f"Available metrics: {metrics}")
    
    # Filter to selected metrics if provided
    if selected_metrics:
        metrics = [m for m in metrics if m in selected_metrics]
        print(f"Using selected metrics: {metrics}")
    
    # Create individual heatmaps for each metric
    for metric in metrics:
        print(f"Generating heatmap for {metric}...")
        plt.figure(figsize=(12, 10))
        
        # Create data matrix for heatmap
        data = []
        for model_term in terms:
            row = []
            for dataset_term in terms:
                if dataset_term in results.get(model_term, {}) and metric in results[model_term][dataset_term]:
                    row.append(results[model_term][dataset_term][metric])
                else:
                    row.append(np.nan)  # Use NaN for missing data
            data.append(row)
            
        # Convert to numpy array
        data_matrix = np.array(data)
        
        # Create heatmap with nan values masked
        ax = sns.heatmap(data_matrix, annot=True, fmt=".3f", cmap="viridis",
                   xticklabels=terms, yticklabels=terms, mask=np.isnan(data_matrix))
        plt.title(f"{metric.replace('_', ' ').title()} - Model (y-axis) vs Dataset (x-axis)")
        plt.ylabel("Model trained on")
        plt.xlabel("Dataset")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{metric}_heatmap.png", dpi=300, bbox_inches='tight')
        
        # Close to avoid memory issues
        plt.close()
        
    # Create specialized AUC ROC heatmap with different styling
    if "auc_roc" in metrics:
        print("Generating specialized AUC ROC heatmap...")
        plt.figure(figsize=(14, 12))
        
        # Create data matrix for AUC ROC heatmap
        auc_data = []
        for model_term in terms:
            row = []
            for dataset_term in terms:
                if dataset_term in results.get(model_term, {}) and "auc_roc" in results[model_term][dataset_term]:
                    # Convert to percentage format (0-100) for readability
                    auc_value = results[model_term][dataset_term]["auc_roc"] * 100
                    row.append(auc_value)
                else:
                    row.append(np.nan)  # Use NaN for missing data
            auc_data.append(row)
            
        # Convert to numpy array
        auc_matrix = np.array(auc_data)
        
        # Create heatmap with different styling and nan values masked
        ax = sns.heatmap(auc_matrix, annot=True, fmt=".1f", cmap="viridis",
                     xticklabels=terms, yticklabels=terms,
                     vmin=50, vmax=100, mask=np.isnan(auc_matrix))  # AUC ranges from 0.5 to 1.0
                     
        # Set title and labels
        plt.title("AUC ROC Scores Across Terms (%)", fontsize=16)
        plt.ylabel("Source Term Dataset", fontsize=14)
        plt.xlabel("Target Term Dataset", fontsize=14)
        
        # Improve tick label formatting
        plt.xticks(rotation=45, ha='right', fontsize=11)
        plt.yticks(fontsize=11)
        
        # Add color bar label
        cbar = ax.collections[0].colorbar
        cbar.set_label("AUC ROC Score (%)", fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/auc_roc_heatmap_styled.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Generate matrix of missing value indicators
    completeness_matrix = np.zeros((len(terms), len(terms)))
    for i, model_term in enumerate(terms):
        for j, dataset_term in enumerate(terms):
            if dataset_term in results.get(model_term, {}) and metrics[0] in results[model_term][dataset_term]:
                completeness_matrix[i, j] = 1.0
    
    plt.figure(figsize=(12, 10))
    # Fix: Change fmt from "d" (integer) to ".0f" (float with 0 decimal places)
    sns.heatmap(completeness_matrix, annot=True, fmt=".0f", cmap="Blues",
               xticklabels=terms, yticklabels=terms)
    plt.title("Data Availability Matrix (1=Available, 0=Missing)")
    plt.ylabel("Model trained on")
    plt.xlabel("Dataset")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/data_availability_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate diagonal vs. off-diagonal comparison (in-domain vs. out-of-domain)
    print("Generating in-domain vs. out-of-domain comparison...")
    
    in_domain = {metric: [] for metric in metrics}
    out_domain = {metric: [] for metric in metrics}
    
    # Collect in-domain and out-of-domain performance for each model
    for model_term in terms:
        for metric in metrics:
            # In-domain is where model_term == dataset_term
            if model_term in results.get(model_term, {}) and model_term in results[model_term]:
                if metric in results[model_term][model_term]:
                    in_domain[metric].append(results[model_term][model_term][metric])
            
            # Out-of-domain is average of all others
            out_values = [results[model_term][dataset_term][metric] 
                         for dataset_term in terms 
                         if dataset_term != model_term 
                         and dataset_term in results.get(model_term, {})
                         and metric in results[model_term][dataset_term]]
            if out_values:
                out_domain[metric].append(np.mean(out_values))
    
    # Create a bar chart for diagonal vs. off-diagonal (in-domain vs. out-of-domain)
    plt.figure(figsize=(12, 8))
    
    # Calculate grand averages
    in_domain_avgs = [np.mean(in_domain[metric]) for metric in metrics]
    out_domain_avgs = [np.mean(out_domain[metric]) for metric in metrics]
    
    # Create a bar chart
    bar_width = 0.35
    x = np.arange(len(metrics))
    
    plt.bar(x - bar_width/2, in_domain_avgs, bar_width, label='In-Domain')
    plt.bar(x + bar_width/2, out_domain_avgs, bar_width, label='Out-of-Domain')
    
    plt.xlabel('Metrics')
    plt.ylabel('Average Value')
    plt.title('In-Domain vs Out-of-Domain Performance')
    plt.xticks(x, [metric.replace('_', ' ').title() for metric in metrics])
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/in_vs_out_domain.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create summary CSV
    print("Generating summary CSV...")
    summary_data = []
    
    # For each model, calculate average metrics
    for model_term in terms:
        row = {"Model": model_term}
        
        # In-domain performance (diagonal)
        if model_term in results.get(model_term, {}) and model_term in results[model_term]:
            for metric in metrics:
                if metric in results[model_term][model_term]:
                    row[f"{metric}_in_domain"] = results[model_term][model_term][metric]
        
        # Out-of-domain performance (average of off-diagonal)
        out_domain_results = {metric: [] for metric in metrics}
        for dataset_term in terms:
            if dataset_term != model_term and dataset_term in results.get(model_term, {}):
                for metric in metrics:
                    if metric in results[model_term][dataset_term]:
                        out_domain_results[metric].append(results[model_term][dataset_term][metric])
        
        # Calculate averages
        for metric in metrics:
            if out_domain_results[metric]:
                row[f"{metric}_out_domain"] = np.mean(out_domain_results[metric])
            else:
                row[f"{metric}_out_domain"] = None
            
            # Calculate drop (in_domain - out_domain)
            if f"{metric}_in_domain" in row and f"{metric}_out_domain" in row and row[f"{metric}_out_domain"] is not None:
                row[f"{metric}_drop"] = row[f"{metric}_in_domain"] - row[f"{metric}_out_domain"]
        
        summary_data.append(row)
    
    # Create dataframe and save
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(f"{output_dir}/cross_term_summary.csv", index=False)
    
    print(f"All visualizations saved to {output_dir}/")

if __name__ == "__main__":
    # Path to the JSON results file
    json_path = "cross_term_evaluation/cross_term_evaluation_results.json"
    
    # Output directory for visualizations
    output_dir = "cross_term_heatmaps"
    
    # Specific metrics to visualize (comment this line to generate all metrics)
    selected_metrics = ["accuracy", "balanced_accuracy", "f1_score", "auc_roc"]
    
    # Load results
    print(f"Loading evaluation results from {json_path}...")
    results = load_evaluation_results(json_path)
    
    # Generate heatmaps and other visualizations
    generate_heatmaps(results, output_dir, selected_metrics)
    
    print("Done!")