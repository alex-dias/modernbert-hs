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

# from density_ensemble_russian import DensityWeightedEnsemble, load_trained_models as load_density_models

class DensityWeightedEnsemble:
    """Density-weighted ensemble classifier."""
    def __init__(self, model_paths, density_weights_path=None, use_median=False, uniform=False):
        self.model_paths = model_paths
        self.models = {}
        self.tokenizers = {}
        self.term_names = list(model_paths.keys())
        
        if uniform:
            print("Using uniform weights for ensemble.")
            self.weights = {term: 1.0/len(self.term_names) for term in self.term_names}
        elif density_weights_path and os.path.exists(density_weights_path):
            # Try to load weights from file
            try:
                # Based on all_term_densities.csv structure: ,asian,black... (term is in columns)
                # Russian is the last row? No, wait.
                # Let's check the CSV format again. 
                # all_term_densities.csv has terms in columns 1 onwards and rows 1 onwards.
                # The last column is 'russian'. 
                # Usually we want the density relative to Russian.
                df_densities = pd.read_csv(density_weights_path, index_col=0)
                print(f"Loaded density weights from {density_weights_path}")
                
                # We want the 'russian' column values (densities of each term to russian)
                # Note: These values might be distances (higher = less dense)
                russian_densities = df_densities['russian']
                
                # If these are distances, we might need to invert or use a kernel.
                # Simple inversion or softmax for weights.
                # Let's assume they are distances based on the high values (1200+).
                # Weight = 1 / distance
                raw_weights = {term: 1.0/russian_densities[term] if term in russian_densities else 0.0 for term in self.term_names}
                
                # Normalize weights
                weight_sum = sum(raw_weights.values())
                self.weights = {k: v/weight_sum if weight_sum > 0 else 1.0/len(self.term_names) for k, v in raw_weights.items()}
                print(f"Normalized density weights: {self.weights}")
            except Exception as e:
                print(f"Error loading weights from {density_weights_path}: {e}. Falling back to uniform.")
                self.weights = {term: 1.0/len(self.term_names) for term in self.term_names}
        else:
            print(f"Warning: Density weights file not found. Using uniform weights.")
            self.weights = {term: 1.0/len(self.term_names) for term in self.term_names}
        
        # Load models
        for term in self.term_names:
            paths = model_paths[term]
            print(f"Loading model for {term}...")
            self.tokenizers[term] = AutoTokenizer.from_pretrained(paths['tokenizer_path'])
            self.models[term] = AutoModelForSequenceClassification.from_pretrained(paths['model_path']).to(device)

    def predict(self, texts, batch_size=32):
        if not isinstance(texts, list):
            texts = [texts]
        
        all_model_scores = {term: [] for term in self.term_names}
        
        for term in self.term_names:
            model = self.models[term]
            tokenizer = self.tokenizers[term]
            model_scores = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=140)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
                    model_scores.extend(probs[:, 1].cpu().numpy())
            
            all_model_scores[term] = np.array(model_scores)
            
        # Weighted average
        model_scores_array = np.array([all_model_scores[term] for term in self.term_names])
        weights_array = np.array([self.weights[term] for term in self.term_names])
        weighted_scores = np.sum(model_scores_array * weights_array[:, np.newaxis], axis=0)
        
        hate_predictions = weighted_scores >= 0.5
        final_predictions = ['hate' if pred else 'no hate' for pred in hate_predictions]
        return final_predictions, weighted_scores.tolist()

class EnsembleClassifier:
    """Standard ensemble classifier (Majority Vote or Weighted Average)."""
    def __init__(self, model_paths, voting="majority", weights=None):
        self.model_paths = model_paths
        self.voting = voting
        self.models = {}
        self.tokenizers = {}
        self.term_names = list(model_paths.keys())
        
        if weights is None:
            self.weights = {term: 1.0/len(model_paths) for term in model_paths.keys()}
        else:
            self.weights = weights
            weight_sum = sum(self.weights.values())
            self.weights = {k: v/weight_sum for k, v in self.weights.items()}
            
        for term, paths in model_paths.items():
            print(f"Loading model for {term}...")
            self.tokenizers[term] = AutoTokenizer.from_pretrained(paths['tokenizer_path'])
            self.models[term] = AutoModelForSequenceClassification.from_pretrained(paths['model_path']).to(device)

    def predict(self, texts, batch_size=32):
        if not isinstance(texts, list):
            texts = [texts]
            
        all_model_scores = {term: [] for term in self.term_names}
        
        for term in self.term_names:
            model = self.models[term]
            tokenizer = self.tokenizers[term]
            model_scores = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=140)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
                    model_scores.extend(probs[:, 1].cpu().numpy())
            
            all_model_scores[term] = np.array(model_scores)
            
        model_scores_array = np.array([all_model_scores[term] for term in self.term_names])
        
        if self.voting == "majority":
            binary_preds = (model_scores_array >= 0.5).astype(int)
            vote_counts = np.sum(binary_preds, axis=0)
            hate_predictions = vote_counts > (len(self.term_names) / 2)
            confidences = vote_counts / len(self.term_names)
        elif self.voting == "weighted_average":
            weights_array = np.array([self.weights[term] for term in self.term_names])
            weighted_scores = np.sum(model_scores_array * weights_array[:, np.newaxis], axis=0)
            hate_predictions = weighted_scores >= 0.5
            confidences = weighted_scores
            
        final_predictions = ['hate' if pred else 'no hate' for pred in hate_predictions]
        return final_predictions, confidences.tolist()

def load_v2_density_models(base_dir="models_knn_tox_refactored/tomh_toxigen_roberta/ensemble_density_ratio_raw_noout"):
    """
    Find and load model paths from the new directory structure.
    """
    if not os.path.exists(base_dir):
        print(f"Error: Base directory {base_dir} not found!")
        return {}
    
    model_paths = {}
    for term in os.listdir(base_dir):
        term_dir = os.path.join(base_dir, term)
        if os.path.isdir(term_dir):
            final_path = os.path.join(term_dir, "final")
            tokenizer_path = os.path.join(term_dir, "tokenizer")
            
            if os.path.exists(final_path) and os.path.exists(tokenizer_path):
                model_paths[term] = {
                    "model_path": final_path,
                    "tokenizer_path": tokenizer_path
                }
    
    return model_paths

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

def load_ensemble_model(ensemble_type="density_avg", ensemble_models_dir="models_knn_tox_refactored/tomh_toxigen_roberta/ensemble_density_ratio_raw_noout", weights_path="all_term_densities.csv", uniform=False):
    """
    Load the density-weighted ensemble model.
    """
    print(f"Loading {ensemble_type} ensemble model (uniform={uniform})...")
    
    # Load model paths from the new directory structure
    model_paths = load_v2_density_models(ensemble_models_dir)
    
    if not model_paths:
        print("Error: No model paths found for ensemble")
        return None
    
    # Determine ensemble type and load appropriate model
    if ensemble_type == "density_avg":
        # Load density-weighted ensemble
        ensemble = DensityWeightedEnsemble(
            model_paths, 
            weights_path,
            use_median=False,
            uniform=uniform
        )
    elif ensemble_type == "majority":
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

def cross_evaluate_terms(terms, ensemble_models_dir="models_knn_tox_refactored/tomh_toxigen_roberta/ensemble_density_ratio_raw_noout", output_dir="cross_term_evaluation", config_name="uniform"):
    """
    Evaluate each term's model on each term's dataset and generate comparison metrics.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Ensure images directory exists
    if not os.path.exists("images"):
        os.makedirs("images")
    
    # Initialize results dictionary
    results = {}
    
    # Dictionary to store datasets
    term_datasets = {}
    
    # Get model paths for individual terms (not using base_dir here, keeping original logic if needed)
    # But wait, original logic used ensemble_models_dir as base. Let's keep it consistent.
    model_paths = {}
    for term in [t for t in terms if t != "russian"]:
        model_path = f"{ensemble_models_dir}/{term}/final"
        tokenizer_path = f"{ensemble_models_dir}/{term}/tokenizer"
        
        if os.path.exists(model_path) and os.path.exists(tokenizer_path):
            model_paths[term] = {
                "model_path": model_path,
                "tokenizer_path": tokenizer_path
            }
        else:
            print(f"Warning: Model paths for {term} not found at {model_path}")
    
    # Preload all datasets
    print("Loading datasets...")
    for term in terms:
        print(f"- Loading dataset for {term}")
        term_datasets[term] = load_term_dataset(term)
    
    # Evaluate Russian ensemble model
    if "russian" in terms:
        print(f"\nEvaluating Russian ensemble model ({config_name})...")
        uniform = (config_name == "uniform")
        ensemble = load_ensemble_model("density_avg", ensemble_models_dir, uniform=uniform)
        
        if ensemble:
            results["russian_ensemble"] = {}
            
            # Test on each dataset
            for dataset_term in terms:
                if dataset_term not in term_datasets:
                    continue
                    
                dataset = term_datasets[dataset_term]
                
                print(f"- Testing on {dataset_term} dataset...")
                predictions, confidences = ensemble.predict(dataset["text"])
                metrics = evaluate_model([1 if p == 'hate' else 0 for p in predictions], dataset["label"], confidences)
                
                # Store results
                results["russian_ensemble"][dataset_term] = metrics
            
            # Free memory
            del ensemble
            torch.cuda.empty_cache()
            gc.collect()
    
    # For each term model
    for model_term in [t for t in terms if t != "russian"]:
        if model_term not in model_paths:
            continue
            
        results[model_term] = {}
        print(f"\nEvaluating model trained on {model_term}...")
        
        model, tokenizer = predict_with_model_loader(model_term, model_paths[model_term])
        
        if model is None:
            continue
        
        for dataset_term in terms:
            if dataset_term not in term_datasets: continue
            dataset = term_datasets[dataset_term]
            print(f"- Testing on {dataset_term} dataset...")
            predictions, confidences = predict_with_model(model, tokenizer, dataset["text"])
            metrics = evaluate_model(predictions, dataset["label"], confidences)
            results[model_term][dataset_term] = metrics
        
        del model
        del tokenizer
        torch.cuda.empty_cache()
        gc.collect()
    
    # Save results
    with open(f"{output_dir}/cross_term_results_{config_name}.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Generate plots
    generate_comparison_plots(results, "images", suffix=config_name)
    
    return results

def predict_with_model_loader(term, paths):
    """Utility to load model from paths."""
    print(f"Loading model for {term}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(paths["tokenizer_path"])
        model = AutoModelForSequenceClassification.from_pretrained(paths["model_path"])
        model.to(device)
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model for {term}: {e}")
        return None, None
    
    return results

def generate_comparison_plots(results, output_dir, suffix=""):
    """
    Generate comparison plots from evaluation results.
    
    Parameters:
    - results: Dictionary of evaluation results
    - output_dir: Directory to save output files
    - suffix: Suffix to add to filenames
    """
    if suffix:
        suffix = f"_{suffix}"
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
        plt.savefig(f"{output_dir}/{metric}_heatmap{suffix}.png", dpi=300, bbox_inches='tight')
        
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
        plt.savefig(f"{output_dir}/auc_roc_heatmap_styled{suffix}.png", dpi=300, bbox_inches='tight')
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
    plt.savefig(f"{output_dir}/average_model_performance{suffix}.png", dpi=300, bbox_inches='tight')
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
    plt.savefig(f"{output_dir}/in_vs_out_domain{suffix}.png", dpi=300, bbox_inches='tight')
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
    summary_df.to_csv(f"{output_dir}/cross_term_summary{suffix}.csv", index=False)
    
    print(f"\nAll plots and data have been saved to {output_dir} directory")

if __name__ == "__main__":
    # Directory containing trained models
    ensemble_models_dir = "models_knn_tox_refactored/tomh_toxigen_roberta/ensemble_density_ratio_raw_noout"
    
    # Output directory for results
    output_dir = "cross_term_evaluation"
    
    # Get list of terms to evaluate
    id_terms = dh.getListOfIdTerms()
    
    print(f"Performing cross-term evaluation for terms: {id_terms}")
    
    # Run cross-evaluation with UNIFORM weights
    print("\n" + "="*50)
    print("RUNNING WITH UNIFORM WEIGHTS")
    print("="*50)
    results_uniform = cross_evaluate_terms(id_terms, ensemble_models_dir, output_dir, config_name="uniform")
    
    # Run cross-evaluation with DENSITY weights
    print("\n" + "="*50)
    print("RUNNING WITH DENSITY WEIGHTS")
    print("="*50)
    results_density = cross_evaluate_terms(id_terms, ensemble_models_dir, output_dir, config_name="density_weighted")
    
    print("Evaluation complete!")