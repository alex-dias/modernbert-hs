import os
import pandas as pd
import numpy as np
import json
import torch
import re
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_recall_fscore_support, confusion_matrix, balanced_accuracy_score

import dataHandler as dh
from density_ensemble_russian import DensityWeightedEnsemble, load_trained_models as load_density_models

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def load_ensemble_models(ensemble_models_dir="modernbert_ensemble"):
    """
    Load ensemble models from the ensemble directory.
    
    Parameters:
    - ensemble_models_dir: Directory containing the ensemble models
    
    Returns:
    - Dictionary mapping model types to their model paths and configurations (not the actual loaded models)
    """
    ensemble_models = {}
    
    # Load ensemble configuration if it exists
    config_path = f"{ensemble_models_dir}/ensemble_config.json"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            ensemble_config = json.load(f)
            
        # Store ensemble configurations
        if "models" in ensemble_config:
            # Always include majority voting model
            ensemble_models["majority"] = {
                "type": "majority",
                "paths": ensemble_config["models"]
            }
            
            # Weighted average with uniform weights
            uniform_weights = {term: 1.0/len(ensemble_config["models"]) for term in ensemble_config["models"].keys()}
            ensemble_models["weighted_avg"] = {
                "type": "weighted_average",
                "paths": ensemble_config["models"],
                "weights": uniform_weights
            }
            
            # Optimized weighted ensemble
            if "optimized_weights" in ensemble_config:
                # Ensure optimized weights are normalized to sum to 1
                opt_weights = ensemble_config["optimized_weights"]
                weights_sum = sum(opt_weights.values())
                normalized_opt_weights = {k: v/weights_sum for k, v in opt_weights.items()}
                
                ensemble_models["optimized"] = {
                    "type": "weighted_average",
                    "paths": ensemble_config["models"],
                    "weights": normalized_opt_weights
                }
    
    # Store paths for density-weighted ensemble
    density_weights_path = "term_densities_to_russian.csv"
    if os.path.exists(density_weights_path):
        model_paths = load_density_models()
        if model_paths:
            ensemble_models["density_avg"] = {
                "type": "density",
                "paths": model_paths,
                "weights_path": density_weights_path,
                "use_median": False
            }
            
            ensemble_models["density_median"] = {
                "type": "density",
                "paths": model_paths,
                "weights_path": density_weights_path,
                "use_median": True
            }
            
            ensemble_models["majority"] = {
                "type": "majority",
                "paths": model_paths,
                "weights_path": density_weights_path,
                "use_median": False
            }
    
    return ensemble_models

def load_knn_models(knn_models_dir="models_knn"):
    """
    Load paths to models from the models_knn directory.
    
    Returns:
        dict: Dictionary mapping model names to their paths (NOT the actual models)
    """
    knn_models = {}
    knn_pattern = re.compile(r"ModernBERT-k(\d+)(?:_balanced)?(?:-classifier|-final)$")
    
    # List directories in models_knn folder
    if not os.path.exists(knn_models_dir):
        print(f"KNN models directory '{knn_models_dir}' not found.")
        return knn_models
        
    for item in os.listdir(knn_models_dir):
        # Check if this is a model directory
        model_path = os.path.join(knn_models_dir, item)
        if os.path.isdir(model_path) and knn_pattern.match(item):
            # Extract k value and determine if balanced
            match = knn_pattern.match(item)
            if match:
                k_value = match.group(1)
                balanced = "_balanced" in item
                final = "-final" in item
                
                # Create a descriptive name
                model_name = f"KNN-k{k_value}{'-balanced' if balanced else ''}"
                
                # For final models, look for the corresponding tokenizer
                if final:
                    tokenizer_path = model_path.replace("-final", "-tokenizer")
                    if os.path.isdir(tokenizer_path):
                        print(f"Found KNN model: {model_name}")
                        knn_models[model_name] = {
                            "model_path": model_path,
                            "tokenizer_path": tokenizer_path
                        }
    
    return knn_models

def evaluate_knn_model(model_name, model_paths, test_dict):
    """
    Evaluate a KNN model on test data.
    
    Parameters:
    - model_name: Name of the model
    - model_paths: Dictionary containing model and tokenizer paths
    - test_dict: Dictionary with 'text' and 'label' keys
    
    Returns:
    - Dictionary of evaluation metrics
    """
    print(f"Evaluating {model_name}...")
    
    # Clear GPU memory before loading model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_paths["tokenizer_path"])
    model = AutoModelForSequenceClassification.from_pretrained(model_paths["model_path"])
    model.to(device)  # Only move to GPU after loading
    
    results = []
    confidences = []
    
    # Process in batches
    batch_size = 16  # Smaller batch size to reduce memory usage
    for i in range(0, len(test_dict['text']), batch_size):
        batch_texts = test_dict['text'][i:i+batch_size]
        
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
            
            # Convert scores to predictions
            batch_preds = ["hate" if p >= 0.5 else "no hate" for p in hate_probs]
            results.extend(batch_preds)
            confidences.extend(hate_probs.tolist())
    
    # Convert predictions and true labels to binary
    pred_binary = [1 if p == 'hate' else 0 for p in results]
    true_binary = [1 if l == 'hate' else 0 for l in test_dict['label']]
    
    # Calculate confusion matrix
    cm = confusion_matrix(true_binary, pred_binary)
    
    # Calculate metrics
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    balanced_acc = (recall + tn/(tn+fp))/2 if (tn+fp) > 0 else recall
    
    # Calculate AUC-ROC
    try:
        aucroc = roc_auc_score(true_binary, confidences)
    except Exception as e:
        print(f"Error calculating AUC-ROC: {e}")
        aucroc = 0.5
    
    metrics = {
        "model": model_name,
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "auc_roc": float(aucroc),
        "confusion_matrix": cm
    }
    
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Balanced Accuracy: {balanced_acc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  AUC-ROC: {aucroc:.4f}")
    
    # Free memory
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    return metrics

def evaluate_ensemble(ensemble_name, ensemble_config, test_dict):
    """
    Evaluate an ensemble model.
    
    Parameters:
    - ensemble_name: Name of the ensemble model
    - ensemble_config: Configuration for the ensemble
    - test_dict: Dictionary with 'text' and 'label' keys
    
    Returns:
    - Dictionary of evaluation metrics
    """
    print(f"Evaluating {ensemble_name} ensemble...")
    
    # Clear GPU memory before loading model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        
    # Load ensemble based on type
    if ensemble_config["type"] == "density":
        # Load density-weighted ensemble
        ensemble = DensityWeightedEnsemble(
            ensemble_config["paths"],
            ensemble_config["weights_path"],
            use_median=ensemble_config["use_median"]
        )
    else:
        # Import here to avoid circular dependency
        from density_ensemble_russian import EnsembleClassifier
        
        # Create appropriate ensemble
        if ensemble_config["type"] == "majority":
            # Majority voting ensemble doesn't need weights
            ensemble = EnsembleClassifier(
                ensemble_config["paths"],
                voting="majority"
            )
        else:
            # For weighted average, ensure weights are normalized
            weights = ensemble_config.get("weights", None)
            if weights:
                # Normalize weights to sum to 1
                weight_sum = sum(weights.values())
                normalized_weights = {k: v/weight_sum for k, v in weights.items()}
            else:
                # Create uniform weights if none provided
                normalized_weights = {term: 1.0/len(ensemble_config["paths"]) for term in ensemble_config["paths"].keys()}
            
            ensemble = EnsembleClassifier(
                ensemble_config["paths"],
                voting="weighted_average",
                weights=normalized_weights
            )
    
    # Get predictions
    predictions, confidences = ensemble.predict(test_dict['text'])
    
    # Convert predictions and true labels to binary
    pred_binary = [1 if p == 'hate' else 0 for p in predictions]
    true_binary = [1 if l == 'hate' else 0 for l in test_dict['label']]
    
    # Calculate confusion matrix
    cm = confusion_matrix(true_binary, pred_binary)
    
    # Calculate metrics
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    balanced_acc = (recall + tn/(tn+fp))/2 if (tn+fp) > 0 else recall
    
    # Calculate AUC-ROC
    try:
        aucroc = roc_auc_score(true_binary, confidences)
    except Exception as e:
        print(f"Error calculating AUC-ROC: {e}")
        aucroc = 0.5
    
    metrics = {
        "model": f"{ensemble_name}_ensemble",
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "auc_roc": float(aucroc),
        "confusion_matrix": cm
    }
    
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Balanced Accuracy: {balanced_acc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  AUC-ROC: {aucroc:.4f}")
    
    # Free memory - remove model references, which should trigger garbage collection
    del ensemble
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    return metrics

def plot_confusion_matrices(metrics_list, output_dir="."):
    """
    Plot confusion matrices for all models.
    
    Parameters:
    - metrics_list: List of metric dictionaries with confusion matrices
    - output_dir: Directory to save the output files
    """
    # Determine grid size
    n_models = len(metrics_list)
    cols = 3
    rows = (n_models + cols - 1) // cols
    
    plt.figure(figsize=(cols * 6, rows * 5))
    
    for i, metrics in enumerate(metrics_list):
        model_name = metrics["model"]
        cm = metrics["confusion_matrix"]
        
        # Add percentages to confusion matrix
        group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
        group_percentages = ["{0:.1%}".format(value) for value in cm.flatten()/np.sum(cm)]
        labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts, group_percentages)]
        labels = np.asarray(labels).reshape(2, 2)
        
        plt.subplot(rows, cols, i + 1)
        sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', xticklabels=['No Hate', 'Hate'], yticklabels=['No Hate', 'Hate'])
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title(f'{model_name}')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/all_models_confusion_matrices.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_comparison_plots(metrics_df, output_dir="."):
    """
    Create comparison bar plots for each metric across all models.
    
    Parameters:
    - metrics_df: DataFrame with metrics for all models
    - output_dir: Directory to save the output files
    """
    metrics_to_compare = ["accuracy", "balanced_accuracy", "precision", "recall", "f1_score", "auc_roc"]
    
    # Sort the dataframe by F1 score descending
    metrics_df = metrics_df.sort_values("f1_score", ascending=False)
    
    # Set a color palette
    model_types = []
    for model in metrics_df['model']:
        if 'KNN' in model:
            model_types.append('KNN')
        elif 'density' in model:
            model_types.append('Density')
        elif 'majority' in model:
            model_types.append('Majority')
        else:
            model_types.append('Weighted')
    
    palette = {'KNN': 'skyblue', 'Weighted': 'lightgreen', 'Density': 'coral', 'Majority': 'mediumpurple'}
    metrics_df['model_type'] = model_types
    
    # Create a figure with subplots for each metric
    plt.figure(figsize=(15, 12))
    
    for i, metric in enumerate(metrics_to_compare):
        plt.subplot(2, 3, i+1)
        
        # Sort by this specific metric for this plot
        metric_df = metrics_df.sort_values(metric, ascending=False)
        
        # Create the plot with custom colors
        ax = sns.barplot(x='model', y=metric, data=metric_df, hue='model_type', 
                         palette=palette, dodge=False)
        
        plt.title(f'Comparison of {metric.replace("_", " ").title()}')
        plt.xticks(rotation=45, ha='right')
        
        # Remove the legend from individual plots (we'll add a single one at the end)
        if i < len(metrics_to_compare) - 1:
            ax.legend_.remove()
            
        # Add value labels on top of each bar
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.3f}', 
                      (p.get_x() + p.get_width() / 2., p.get_height()), 
                      ha = 'center', va = 'bottom',
                      xytext = (0, 5), 
                      textcoords = 'offset points')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/model_metrics_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a radar chart for overall comparison
    plt.figure(figsize=(12, 10))
    
    # Take top 5 models based on F1 score
    top_models = metrics_df.head(5)['model'].tolist()
    top_metrics = metrics_df[metrics_df['model'].isin(top_models)]
    
    # Prepare data for radar chart
    categories = metrics_to_compare
    N = len(categories)
    
    # Create angle values for each metric
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Initialize the plot
    ax = plt.subplot(111, polar=True)
    
    # Draw one axis per variable and add labels
    plt.xticks(angles[:-1], categories, size=12)
    
    # Draw the y-axis labels (0-1 scale)
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], size=10)
    plt.ylim(0, 1)
    
    # Plot each model
    for i, model in enumerate(top_models):
        model_data = top_metrics[top_metrics['model'] == model]
        values = model_data[categories].values.flatten().tolist()
        values += values[:1]  # Close the loop
        
        # Plot values
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=model)
        ax.fill(angles, values, alpha=0.1)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title("Top Models Performance Comparison", size=15)
    
    plt.savefig(f"{output_dir}/top_models_radar_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """
    Main function to run the comparison.
    """
    output_dir = "model_comparisons"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get test data
    print("Loading test data...")
    test_dict = dh.getAnnotadedRussTest()
    print(f"Loaded {len(test_dict['text'])} test samples")
    
    # Load ensemble model paths and configurations
    print("\nLoading ensemble model configurations...")
    ensemble_models = load_ensemble_models()
    print(f"Found {len(ensemble_models)} ensemble model configurations")
    
    # Verify majority voting model is included
    if "majority" not in ensemble_models:
        print("Warning: Majority voting model is not included in the configurations.")
    else:
        print("Confirmed: Majority voting model is included.")
    
    # Load KNN model paths
    print("\nLoading KNN model paths...")
    knn_model_paths = load_knn_models()
    print(f"Found {len(knn_model_paths)} KNN models")
    
    # Evaluate all models
    all_metrics = []
    
    # Evaluate ensemble models
    for model_name, config in ensemble_models.items():
        metrics = evaluate_ensemble(model_name, config, test_dict)
        all_metrics.append(metrics)
    
    # Evaluate KNN models
    for model_name, paths in knn_model_paths.items():
        metrics = evaluate_knn_model(model_name, paths, test_dict)
        all_metrics.append(metrics)
    
    # Convert to DataFrame
    metrics_df = pd.DataFrame([
        {k: v for k, v in m.items() if k != "confusion_matrix"} 
        for m in all_metrics
    ])
    
    # Save as CSV
    metrics_df.to_csv(f"{output_dir}/all_models_metrics.csv", index=False)
    print(f"\nMetrics saved to {output_dir}/all_models_metrics.csv")
    
    # Plot confusion matrices
    plot_confusion_matrices(all_metrics, output_dir)
    print(f"Confusion matrices saved to {output_dir}/all_models_confusion_matrices.png")
    
    # Create comparison plots
    create_comparison_plots(metrics_df, output_dir)
    print(f"Comparison plots saved to {output_dir}/model_metrics_comparison.png and {output_dir}/top_models_radar_comparison.png")
    
    # Print summary
    print("\nTop 5 models by F1 score:")
    top_models = metrics_df.sort_values("f1_score", ascending=False).head(5)
    print(top_models[["model", "f1_score", "accuracy", "balanced_accuracy"]])
    
    # Count of models by type
    model_type_counts = metrics_df['model'].apply(lambda x: 
                                                'KNN' if 'KNN' in x else 
                                                'Majority' if 'majority' in x else
                                                'Density' if 'density' in x else 'Weighted').value_counts()
    print("\nModel types in comparison:")
    print(model_type_counts)

if __name__ == "__main__":
    main()