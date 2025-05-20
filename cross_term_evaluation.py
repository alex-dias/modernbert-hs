import os
import pandas as pd
import numpy as np
import torch
import json
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_recall_fscore_support, confusion_matrix, balanced_accuracy_score
import gc

import dataHandler as dh
from density_ensemble_russian import DensityWeightedEnsemble, load_trained_models as load_density_models

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def load_term_model(term, ensemble_models_dir="modernbert_ensemble"):
    """
    Load a model for a specific identity term.
    
    Parameters:
    - term: The identity term whose model we want to load
    - ensemble_models_dir: Directory containing the trained models
    
    Returns:
    - model: The loaded model
    - tokenizer: The loaded tokenizer
    """
    model_path = f"{ensemble_models_dir}/{term}-final"
    tokenizer_path = f"{ensemble_models_dir}/{term}-tokenizer"
    
    if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
        print(f"Error: Model or tokenizer for term '{term}' not found!")
        return None, None
    
    print(f"Loading model for {term}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    
    return model, tokenizer

def load_ensemble_model(ensemble_type="density_avg", ensemble_models_dir="modernbert_ensemble"):
    """
    Load the density-weighted ensemble model.
    
    Parameters:
    - ensemble_type: Type of ensemble model to load ("density_avg", "density_median", "majority")
    - ensemble_models_dir: Directory containing the ensemble models
    
    Returns:
    - ensemble: The loaded ensemble model
    """
    print(f"Loading {ensemble_type} ensemble model...")
    
    # Load model paths from density_ensemble_russian module
    model_paths = load_density_models()
    
    if not model_paths:
        print("Error: No model paths found for ensemble")
        return None
    
    # Determine ensemble type and load appropriate model
    if ensemble_type == "density_avg":
        # Load density-weighted ensemble using average density values
        ensemble = DensityWeightedEnsemble(
            model_paths, 
            "term_densities_to_russian.csv",
            use_median=False
        )
    elif ensemble_type == "density_median":
        # Load density-weighted ensemble using median density values
        ensemble = DensityWeightedEnsemble(
            model_paths, 
            "term_densities_to_russian.csv",
            use_median=True
        )
    elif ensemble_type == "majority":
        # Import here to avoid circular dependency
        from density_ensemble_russian import EnsembleClassifier
        
        # Create majority voting ensemble
        ensemble = EnsembleClassifier(
            model_paths,
            voting="majority"
        )
    else:
        print(f"Error: Unknown ensemble type '{ensemble_type}'")
        return None
    
    return ensemble

def predict_with_model(model, tokenizer, texts, batch_size=32):
    """
    Make predictions using a model.
    
    Parameters:
    - model: The model to use for predictions
    - tokenizer: The tokenizer to use for processing texts
    - texts: List of texts to classify
    - batch_size: Batch size for processing
    
    Returns:
    - predictions: Binary labels (0 or 1)
    - confidences: Confidence scores for the positive class
    """
    if not isinstance(texts, list):
        texts = [texts]
    
    model_scores = []
    
    # Process texts in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=140)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1)
            
            # Get probability of hate class (index 1)
            hate_probs = probs[:, 1].cpu().numpy()
            model_scores.extend(hate_probs)
    
    # Convert scores to binary predictions and return
    predictions = [1 if score >= 0.5 else 0 for score in model_scores]
    return predictions, model_scores

def load_term_dataset(term):
    """
    Load the test dataset for a specific identity term.
    
    Parameters:
    - term: The identity term whose dataset we want to load
    
    Returns:
    - Dictionary with text samples and labels
    """
    if term == "russian":
        return dh.getAnnotadedRussTest()
    
    # For other terms, get dataset from toxigenDataset
    dataset = dh.toxigenDataset(term)
    return {
        "text": dataset["test"]["text"],
        "label": dataset["test"]["label"]
    }

def evaluate_model(predictions, true_labels, confidences=None):
    """
    Evaluate model predictions against true labels.
    
    Parameters:
    - predictions: List of predicted binary labels (0 or 1)
    - true_labels: List of true binary labels (0 or 1)
    - confidences: List of confidence scores for the positive class (needed for AUC ROC)
    
    Returns:
    - Dictionary of metrics
    """
    # Convert string labels to integers if needed
    if isinstance(predictions[0], str):
        predictions = [1 if p == 'hate' else 0 for p in predictions]
    
    if isinstance(true_labels[0], str):
        true_labels = [1 if l == 'hate' else 0 for l in true_labels]
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='binary')
    balanced_acc = balanced_accuracy_score(true_labels, predictions)
    
    # Calculate AUC ROC if confidence scores are provided
    auc_roc = None
    if confidences is not None:
        # Check if there are both positive and negative samples
        unique_labels = np.unique(true_labels)
        if len(unique_labels) > 1:
            try:
                auc_roc = roc_auc_score(true_labels, confidences)
            except:
                # Fallback if AUC ROC calculation fails
                auc_roc = 0.5  # Default: no discrimination
        else:
            # If all samples are from the same class, AUC is undefined
            auc_roc = 0.5  # Default: no discrimination
    
    # Calculate confusion matrix elements
    tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()
    
    # Return metrics dictionary
    result = {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn)
    }
    
    if auc_roc is not None:
        result["auc_roc"] = float(auc_roc)
        
    return result

def cross_evaluate_terms(terms, ensemble_models_dir="modernbert_ensemble", output_dir="cross_term_evaluation"):
    """
    Evaluate each term's model on each term's dataset and generate comparison metrics.
    Memory-efficient version that loads models only when needed.
    
    Parameters:
    - terms: List of terms to evaluate
    - ensemble_models_dir: Directory containing the trained models
    - output_dir: Directory to save output files
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize results dictionary
    results = {}
    
    # Dictionary to store datasets (pre-loading all datasets to save time)
    term_datasets = {}
    
    # Get model paths (not loading models yet)
    model_paths = {}
    for term in [t for t in terms if t != "russian"]:
        model_path = f"{ensemble_models_dir}/{term}-final"
        tokenizer_path = f"{ensemble_models_dir}/{term}-tokenizer"
        
        if os.path.exists(model_path) and os.path.exists(tokenizer_path):
            model_paths[term] = {
                "model_path": model_path,
                "tokenizer_path": tokenizer_path
            }
        else:
            print(f"Warning: Model paths for {term} not found")
    
    # Preload all datasets
    print("Loading datasets...")
    for term in terms:
        print(f"- Loading dataset for {term}")
        term_datasets[term] = load_term_dataset(term)
    
    # Evaluate Russian ensemble model first
    if "russian" in terms:
        print("\nEvaluating Russian ensemble model...")
        ensemble = load_ensemble_model("density_avg")
        
        if ensemble:
            results["russian_ensemble"] = {}
            
            # Test on each dataset
            for dataset_term in terms:
                if dataset_term not in term_datasets:
                    print(f"Skipping dataset {dataset_term} as it couldn't be loaded.")
                    continue
                    
                dataset = term_datasets[dataset_term]
                
                print(f"- Testing on {dataset_term} dataset...")
                predictions, confidences = ensemble.predict(dataset["text"])
                metrics = evaluate_model([1 if p == 'hate' else 0 for p in predictions], dataset["label"], confidences)
                
                # Store results
                results["russian_ensemble"][dataset_term] = metrics
                
                # Print some metrics
                print(f"  - Accuracy: {metrics['accuracy']:.4f}")
                print(f"  - F1 Score: {metrics['f1_score']:.4f}")
                print(f"  - Balanced Acc: {metrics['balanced_accuracy']:.4f}")
                if 'auc_roc' in metrics:
                    print(f"  - AUC ROC: {metrics['auc_roc']:.4f}")
            
            # Free memory
            del ensemble
            torch.cuda.empty_cache()
            gc.collect()
    
    # For each term model, load it, evaluate on all datasets, then unload it
    for model_term in [t for t in terms if t != "russian"]:
        if model_term not in model_paths:
            print(f"Skipping model {model_term} as it couldn't be found.")
            continue
            
        # Initialize results for this model
        results[model_term] = {}
        
        print(f"\nEvaluating model trained on {model_term}...")
        
        # Load model for this term
        model, tokenizer = load_term_model(model_term, ensemble_models_dir)
        
        if model is None or tokenizer is None:
            print(f"Error: Could not load model for {model_term}.")
            continue
        
        # Test on each dataset
        for dataset_term in terms:
            if dataset_term not in term_datasets:
                print(f"Skipping dataset {dataset_term} as it couldn't be loaded.")
                continue
                
            dataset = term_datasets[dataset_term]
            
            print(f"- Testing on {dataset_term} dataset...")
            predictions, confidences = predict_with_model(model, tokenizer, dataset["text"])
            metrics = evaluate_model(predictions, dataset["label"], confidences)
            
            # Store results
            results[model_term][dataset_term] = metrics
            
            # Print some metrics
            print(f"  - Accuracy: {metrics['accuracy']:.4f}")
            print(f"  - F1 Score: {metrics['f1_score']:.4f}")
            print(f"  - Balanced Acc: {metrics['balanced_accuracy']:.4f}")
            if 'auc_roc' in metrics:
                print(f"  - AUC ROC: {metrics['auc_roc']:.4f}")
        
        # Free memory after evaluating all datasets with this model
        print(f"Unloading model for {model_term} to free memory...")
        del model
        del tokenizer
        torch.cuda.empty_cache()
        gc.collect()
    
    # Save results to file
    with open(f"{output_dir}/cross_term_evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Generate plots
    generate_comparison_plots(results, output_dir)
    
    return results

def generate_comparison_plots(results, output_dir):
    """
    Generate comparison plots from evaluation results.
    
    Parameters:
    - results: Dictionary of evaluation results
    - output_dir: Directory to save output files
    """
    metrics_to_plot = ["accuracy", "balanced_accuracy", "precision", "recall", "f1_score", "auc_roc"]
    terms = list(results.keys())
    
    # Create individual heatmaps for each metric
    for metric in metrics_to_plot:
        # Skip if metric not in results
        if not all(metric in results.get(model_term, {}).get(dataset_term, {}) 
                  for model_term in terms 
                  for dataset_term in terms 
                  if dataset_term in results.get(model_term, {})):
            print(f"Skipping {metric} heatmap as not all models have this metric")
            continue
            
        plt.figure(figsize=(12, 10))
        
        # Create data matrix for heatmap
        data = []
        for model_term in terms:
            row = []
            for dataset_term in terms:
                if dataset_term in results.get(model_term, {}) and metric in results[model_term][dataset_term]:
                    row.append(results[model_term][dataset_term][metric])
                else:
                    row.append(0.0)  # Use 0 for missing data
            data.append(row)
            
        # Convert to numpy array
        data_matrix = np.array(data)
        
        # Create heatmap
        sns.heatmap(data_matrix, annot=True, fmt=".3f", cmap="viridis",
                   xticklabels=terms, yticklabels=terms)
        plt.title(f"{metric.replace('_', ' ').title()} - Model (y-axis) vs Dataset (x-axis)")
        plt.ylabel("Model trained on")
        plt.xlabel("Dataset")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{metric}_heatmap.png", dpi=300, bbox_inches='tight')
        
        # Close to avoid memory issues
        plt.close()
    
    # Create specialized AUC ROC heatmap with different styling
    if "auc_roc" in metrics_to_plot:
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
                    row.append(50.0)  # 0.5 (no discrimination) in percentage
            auc_data.append(row)
            
        # Convert to numpy array
        auc_matrix = np.array(auc_data)
        
        # Create heatmap with different styling
        ax = sns.heatmap(auc_matrix, annot=True, fmt=".1f", cmap="viridis",
                         xticklabels=terms, yticklabels=terms,
                         vmin=50, vmax=100)  # AUC ranges from 0.5 to 1.0
                         
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
    
    # Create bar chart comparison
    # For each model, how well does it perform across all datasets
    plt.figure(figsize=(14, 8))
    
    bar_data = {}
    for metric in metrics_to_plot:
        bar_data[metric] = []
        
        for model_term in terms:
            # Calculate average metric value across all datasets
            metric_values = [results[model_term][dataset_term][metric] 
                            for dataset_term in terms 
                            if dataset_term in results.get(model_term, {}) 
                            and metric in results[model_term][dataset_term]]
            
            if metric_values:
                bar_data[metric].append(np.mean(metric_values))
            else:
                bar_data[metric].append(0.0)
    
    # Set up the bar chart
    bar_width = 0.15
    x = np.arange(len(terms))
    
    # Create bars for each metric
    for i, metric in enumerate(metrics_to_plot):
        plt.bar(x + i*bar_width, bar_data[metric], width=bar_width, 
               label=metric.replace('_', ' ').title())
    
    plt.xlabel('Models')
    plt.ylabel('Average Metric Value')
    plt.title('Average Performance of Each Model Across All Datasets')
    plt.xticks(x + bar_width * (len(metrics_to_plot) - 1) / 2, terms, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/average_model_performance.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create diagonal vs off-diagonal comparison (in-domain vs out-of-domain performance)
    plt.figure(figsize=(12, 8))
    
    in_domain = {metric: [] for metric in metrics_to_plot}
    out_domain = {metric: [] for metric in metrics_to_plot}
    
    # Calculate average in-domain and out-of-domain performance for each model
    for model_term in terms:
        for metric in metrics_to_plot:
            # In-domain is where model_term == dataset_term
            if model_term in results.get(model_term, {}) and dataset_term in results[model_term]:
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
    
    # Calculate grand averages
    in_domain_avgs = [np.mean(in_domain[metric]) for metric in metrics_to_plot]
    out_domain_avgs = [np.mean(out_domain[metric]) for metric in metrics_to_plot]
    
    # Create a bar chart
    bar_width = 0.35
    x = np.arange(len(metrics_to_plot))
    
    plt.bar(x - bar_width/2, in_domain_avgs, bar_width, label='In-Domain')
    plt.bar(x + bar_width/2, out_domain_avgs, bar_width, label='Out-of-Domain')
    
    plt.xlabel('Metrics')
    plt.ylabel('Average Value')
    plt.title('In-Domain vs Out-of-Domain Performance')
    plt.xticks(x, [metric.replace('_', ' ').title() for metric in metrics_to_plot])
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/in_vs_out_domain.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create comparison dataframe and save to CSV
    summary_data = []
    
    # For each model, calculate average metrics
    for model_term in terms:
        row = {"Model": model_term}
        
        # In-domain performance (diagonal)
        if model_term in results.get(model_term, {}) and model_term in results[model_term]:
            for metric in metrics_to_plot:
                if metric in results[model_term][model_term]:
                    row[f"{metric}_in_domain"] = results[model_term][model_term][metric]
        
        # Out-of-domain performance (average of off-diagonal)
        out_domain_results = {metric: [] for metric in metrics_to_plot}
        for dataset_term in terms:
            if dataset_term != model_term and dataset_term in results.get(model_term, {}):
                for metric in metrics_to_plot:
                    if metric in results[model_term][dataset_term]:
                        out_domain_results[metric].append(results[model_term][dataset_term][metric])
        
        # Calculate averages
        for metric in metrics_to_plot:
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
    
    print(f"\nAll plots and data have been saved to {output_dir} directory")

if __name__ == "__main__":
    # Directory containing trained models
    ensemble_models_dir = "modernbert_ensemble"
    
    # Output directory for results
    output_dir = "cross_term_evaluation"
    
    # Get list of terms to evaluate
    id_terms = dh.getListOfIdTerms()
    
    # Optional: Use only a subset of terms for quicker evaluation
    # id_terms = id_terms[:5]  # Uncomment to use only the first 5 terms
    
    print(f"Performing cross-term evaluation for terms: {id_terms}")
    
    # Run cross-evaluation
    results = cross_evaluate_terms(id_terms, ensemble_models_dir, output_dir)
    
    print("Evaluation complete!")