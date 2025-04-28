import os
import pandas as pd
import numpy as np
import torch
import json
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_recall_fscore_support, confusion_matrix, balanced_accuracy_score

import dataHandler as dh

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class DensityWeightedEnsemble:
    def __init__(self, model_paths, density_weights_path, use_median=False):
        """
        Initialize the density-weighted ensemble classifier.
        
        Parameters:
        - model_paths: Dictionary mapping term names to dictionaries with 'model_path' and 'tokenizer_path'
        - density_weights_path: Path to CSV file containing term density values for Russian hate speech
        - use_median: Whether to use median values instead of average for weights
        """
        self.model_paths = model_paths
        self.models = {}
        self.tokenizers = {}
        self.pipelines = {}
        
        # Load density weights from CSV
        df_densities = pd.read_csv(density_weights_path)
        print(f"Loaded density weights from {density_weights_path}")
        
        # Select column to use for weighting (average or median)
        weight_column = 'median' if use_median else 'average'
        
        # Create a dictionary of weights from the dataframe
        self.weights = {}
        for _, row in df_densities.iterrows():
            term = row['term']
            weight = row[weight_column]
            self.weights[term] = weight
            
        # Only keep weights for terms that have models
        self.term_names = [term for term in model_paths.keys() if term in self.weights]
        self.weights = {term: self.weights[term] for term in self.term_names}
        
        # Normalize weights to sum to 1
        weight_sum = sum(self.weights.values())
        self.weights = {k: v/weight_sum for k, v in self.weights.items()}
        
        print(f"Normalized density weights: {self.weights}")
        
        # Load models and tokenizers
        for term in self.term_names:
            paths = model_paths[term]
            print(f"Loading model for {term}...")
            self.tokenizers[term] = AutoTokenizer.from_pretrained(paths['tokenizer_path'])
            self.models[term] = AutoModelForSequenceClassification.from_pretrained(paths['model_path'])
            self.models[term].to(device)
            
            # Create pipeline
            self.pipelines[term] = pipeline(
                task="text-classification",
                tokenizer=self.tokenizers[term],
                model=self.models[term],
                device=0 if torch.cuda.is_available() else -1
            )
    
    def predict(self, texts):
        """
        Make predictions using the density-weighted ensemble.
        
        Parameters:
        - texts: List of strings, texts to classify
        
        Returns:
        - predictions: List of prediction labels
        - confidences: List of confidence scores
        """
        if not isinstance(texts, list):
            texts = [texts]
        
        # Determine batch size based on available memory
        batch_size = 32
        
        # Process in batches for better performance
        all_model_scores = {term: [] for term in self.term_names}
        
        # Process each term's model in batches
        for term in self.term_names:
            model = self.models[term]
            tokenizer = self.tokenizers[term]
            
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
            
            all_model_scores[term] = np.array(model_scores)
        
        # Apply weighted average ensemble method
        final_predictions = []
        final_confidences = []
        
        # Convert to numpy arrays for vectorized operations
        model_scores_array = np.array([all_model_scores[term] for term in self.term_names])
        
        # Apply weights to each model's predictions
        weights_array = np.array([self.weights[term] for term in self.term_names])
        
        # Weighted sum of scores (weights_array[:, np.newaxis] broadcasts weights across all samples)
        weighted_scores = np.sum(model_scores_array * weights_array[:, np.newaxis], axis=0)
        
        # Classify based on threshold
        hate_predictions = weighted_scores >= 0.5
        
        # Confidence is max of score or 1-score
        confidences = np.maximum(weighted_scores, 1 - weighted_scores)
        
        # Convert to strings and list
        final_predictions = ["hate" if pred else "no hate" for pred in hate_predictions]
        final_confidences = confidences.tolist()
        
        return final_predictions, final_confidences

class EnsembleClassifier:
    def __init__(self, model_paths, voting="majority", weights=None):
        """
        Initialize the ensemble classifier.
        
        Parameters:
        - model_paths: Dictionary mapping term names to dictionaries with 'model_path' and 'tokenizer_path'
        - voting: String, voting mechanism to use ('majority', 'weighted_average')
        - weights: Optional dictionary mapping term names to weights for weighted averaging
        """
        self.model_paths = model_paths
        self.voting = voting
        self.models = {}
        self.tokenizers = {}
        self.pipelines = {}
        self.term_names = list(model_paths.keys())
        
        # Default to equal weights if none provided
        if weights is None:
            self.weights = {term: 1.0/len(model_paths) for term in model_paths.keys()}
        else:
            self.weights = weights
            # Normalize weights to sum to 1
            weight_sum = sum(self.weights.values())
            self.weights = {k: v/weight_sum for k, v in self.weights.items()}
        
        # Load models and tokenizers
        for term, paths in model_paths.items():
            print(f"Loading model for {term}...")
            self.tokenizers[term] = AutoTokenizer.from_pretrained(paths['tokenizer_path'])
            self.models[term] = AutoModelForSequenceClassification.from_pretrained(paths['model_path'])
            self.models[term].to(device)
            
            # Create pipeline
            self.pipelines[term] = pipeline(
                task="text-classification",
                tokenizer=self.tokenizers[term],
                model=self.models[term],
                device=0 if torch.cuda.is_available() else -1
            )
    
    def predict(self, texts):
        """
        Make predictions using the ensemble.
        
        Parameters:
        - texts: List of strings, texts to classify
        
        Returns:
        - predictions: List of prediction labels
        - confidences: List of confidence scores
        """
        if not isinstance(texts, list):
            texts = [texts]
        
        # Determine batch size based on available memory
        batch_size = 32
        
        # Process in batches for better performance
        all_model_scores = {term: [] for term in self.term_names}
        
        # Process each term's model in batches
        for term in self.term_names:
            model = self.models[term]
            tokenizer = self.tokenizers[term]
            
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
            
            all_model_scores[term] = np.array(model_scores)
        
        # Apply ensemble method
        final_predictions = []
        final_confidences = []
        
        # Convert to numpy arrays for vectorized operations
        model_scores_array = np.array([all_model_scores[term] for term in self.term_names])
        
        if self.voting == "majority":
            # Convert scores to binary predictions (1 for hate, 0 for no hate)
            binary_preds = (model_scores_array >= 0.5).astype(int)
            
            # Sum across models (axis 0) to get votes for each text
            vote_counts = np.sum(binary_preds, axis=0)
            
            # Majority rule
            hate_predictions = vote_counts > (len(self.term_names) / 2)
            
            # Calculate confidence as proportion of votes
            confidences = np.maximum(vote_counts, len(self.term_names) - vote_counts) / len(self.term_names)
            
            # Convert to strings and list
            final_predictions = ["hate" if pred else "no hate" for pred in hate_predictions]
            final_confidences = confidences.tolist()
            
        elif self.voting == "weighted_average":
            # Apply weights to each model's predictions
            weights_array = np.array([self.weights[term] for term in self.term_names])
            
            # Weighted sum of scores (weights_array[:, np.newaxis] broadcasts weights across all samples)
            weighted_scores = np.sum(model_scores_array * weights_array[:, np.newaxis], axis=0)
            
            # Classify based on threshold
            hate_predictions = weighted_scores >= 0.5
            
            # Confidence is max of score or 1-score
            confidences = np.maximum(weighted_scores, 1 - weighted_scores)
            
            # Convert to strings and list
            final_predictions = ["hate" if pred else "no hate" for pred in hate_predictions]
            final_confidences = confidences.tolist()
        
        return final_predictions, final_confidences

def load_trained_models(ensemble_models_dir="modernbert_ensemble"):
    """
    Load trained models from the ensemble directory.
    
    Parameters:
    - ensemble_models_dir: Directory containing the trained models
    
    Returns:
    - Dictionary mapping term names to model and tokenizer paths
    """
    # Try to load from ensemble_config.json if it exists
    config_path = f"{ensemble_models_dir}/ensemble_config.json"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        return config["models"]
    
    # Otherwise, construct paths manually
    trained_models = {}
    id_terms = dh.getListOfIdTerms()[:-1]  # Exclude 'russian'
    
    for term in id_terms:
        model_path = f"{ensemble_models_dir}/{term}-final"
        tokenizer_path = f"{ensemble_models_dir}/{term}-tokenizer"
        
        if os.path.exists(model_path) and os.path.exists(tokenizer_path):
            trained_models[term] = {
                "model_path": model_path,
                "tokenizer_path": tokenizer_path
            }
            print(f"Found existing model for {term}")
    
    return trained_models

def evaluate_model(ensemble, test_dict, output_dir="modernbert_ensemble"):
    """
    Evaluate the model on test data and plot confusion matrix.
    
    Parameters:
    - ensemble: The ensemble model to evaluate
    - test_dict: Dictionary containing test data and labels
    - output_dir: Directory to save output files
    
    Returns:
    - Dictionary of evaluation metrics
    """
    predictions, confidences = ensemble.predict(test_dict['text'])
    
    # Convert string predictions and true labels to binary
    pred_binary = [1 if p == 'hate' else 0 for p in predictions]
    true_binary = [1 if l == 'hate' else 0 for l in test_dict['label']]
    
    # Calculate confusion matrix
    cm = confusion_matrix(true_binary, pred_binary)
    
    # Add percentages to confusion matrix
    group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
    group_percentages = ["{0:.1%}".format(value) for value in cm.flatten()/np.sum(cm)]
    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', xticklabels=['No Hate', 'Hate'], yticklabels=['No Hate', 'Hate'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix - Density Weighted Ensemble')
    plt.savefig(f"{output_dir}/density_weighted_ensemble_confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.show()
    
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
    except:
        aucroc = 0.5
    
    metrics = {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "auc_roc": float(aucroc)
    }
    
    print(f"Metrics for Density Weighted Ensemble:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Balanced Accuracy: {balanced_acc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  AUC-ROC: {aucroc:.4f}")
    
    # Save metrics to a file
    with open(f"{output_dir}/density_weighted_ensemble_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    return metrics

def compare_with_standard_ensemble(density_metrics, ensemble_models_dir="modernbert_ensemble", use_median=False):
    """
    Compare density-weighted ensemble with standard ensemble approaches.
    
    Parameters:
    - density_metrics: Metrics from density-weighted ensemble
    - ensemble_models_dir: Directory containing ensemble models
    - use_median: Whether median density was used instead of average
    """
    # Try to load previously saved metrics if they exist
    standard_metrics = {}
    
    if os.path.exists(f"{ensemble_models_dir}/majority_ensemble_metrics.json"):
        with open(f"{ensemble_models_dir}/majority_ensemble_metrics.json", "r") as f:
            standard_metrics["majority"] = json.load(f)
    
    if os.path.exists(f"{ensemble_models_dir}/weighted_average_ensemble_metrics.json"):
        with open(f"{ensemble_models_dir}/weighted_average_ensemble_metrics.json", "r") as f:
            standard_metrics["weighted_average"] = json.load(f)
            
    # Add density-weighted metrics
    weight_type = "median" if use_median else "average"
    standard_metrics[f"density_{weight_type}"] = density_metrics
    
    # Compare metrics if we have at least one other ensemble to compare with
    if len(standard_metrics) > 1:
        metrics_to_compare = ["accuracy", "balanced_accuracy", "f1_score", "auc_roc"]
        
        # Create a dataframe for comparison
        comparison_data = []
        for ensemble_name, metrics in standard_metrics.items():
            row = {"Ensemble": ensemble_name}
            for metric in metrics_to_compare:
                if metric in metrics:
                    row[metric] = metrics[metric]
            comparison_data.append(row)
            
        comparison_df = pd.DataFrame(comparison_data)
        print("\nEnsemble Comparison:")
        print(comparison_df)
        
        # Save comparison to CSV
        comparison_df.to_csv(f"{ensemble_models_dir}/ensemble_comparison.csv", index=False)
        
        # Create bar plots for key metrics
        plt.figure(figsize=(12, 8))
        for i, metric in enumerate(metrics_to_compare):
            plt.subplot(2, 2, i+1)
            sns.barplot(x="Ensemble", y=metric, data=comparison_df)
            plt.title(f"Comparison of {metric}")
            plt.xticks(rotation=45)
            plt.tight_layout()
        
        plt.savefig(f"{ensemble_models_dir}/ensemble_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    return standard_metrics

if __name__ == "__main__":
    # Directory containing trained models
    ensemble_models_dir = "modernbert_ensemble"
    
    # Path to density weights CSV
    density_weights_path = "term_densities_to_russian.csv"
    
    # Whether to use median instead of average for density weights
    use_median = False  # Set to True to use median values
    
    print(f"Loading trained models from {ensemble_models_dir}...")
    trained_models = load_trained_models(ensemble_models_dir)
    print(f"Loaded {len(trained_models)} trained models")
    
    # Create density-weighted ensemble
    print(f"Creating density-weighted ensemble using {'median' if use_median else 'average'} densities...")
    density_ensemble = DensityWeightedEnsemble(
        trained_models, 
        density_weights_path,
        use_median=use_median
    )
    
    # Load Russian test dataset
    print("Loading Russian test dataset...")
    russian_test_data = dh.getAnnotadedRussTest()
    print(f"Loaded {len(russian_test_data['text'])} test samples")
    
    # Evaluate the ensemble
    print("Evaluating density-weighted ensemble...")
    metrics = evaluate_model(density_ensemble, russian_test_data, ensemble_models_dir)
    
    # Compare with other ensemble methods
    print("Comparing with other ensemble methods...")
    comparison = compare_with_standard_ensemble(metrics, ensemble_models_dir, use_median)
    
    print("Done!")