import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, logging
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from sklearn.metrics import balanced_accuracy_score
import numpy as np
from dataHandler import getAnnotadedRussTest
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
import os

# Set less verbose logging
logging.set_verbosity_error()

def evaluate_toxigen_model():
    print("Loading ToxiGen model and tokenizer...")    # Load the ToxiGen model from Hugging Face
    model_name = "tomh/toxigen_roberta"  # This is the ToxiGen RoBERTa model
    
    # Add options to prioritize GPU and avoid warning messages
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print(f"Model loaded. Using device: {device}")
      # Get the annotated Russian test dataset
    print("Loading annotated Russian test dataset...")
    test_data = getAnnotadedRussTest()
    texts = test_data["text"]
    
    # Ensure labels are consistently numeric (convert strings to integers if needed)
    true_labels = []
    for label in test_data["label"]:
        if isinstance(label, str):
            # Convert string labels to int (assuming 'hate' = 1, anything else = 0)
            true_labels.append(1 if label.lower() == 'hate' else 0)
        else:
            # Already numeric
            true_labels.append(int(label))
    
    print(f"Dataset loaded: {len(texts)} samples")
    print(f"Label types cleaned: all converted to integers")
      # Tokenize and prepare predictions
    predictions = []
    probabilities = []
    
    print("Making predictions...")
    # Process in batches to avoid memory issues
    batch_size = 32 if torch.cuda.is_available() else 16  # Larger batch size if GPU is available
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            batch_preds = torch.argmax(probs, dim=1).cpu().numpy()
            batch_probs = probs[:, 1].cpu().numpy()  # Probability of "hate" class
        
        predictions.extend(batch_preds)
        probabilities.extend(batch_probs)
        
        # Print progress
        if (i + batch_size) % 100 == 0 or (i + batch_size) >= len(texts):
            print(f"Processed {min(i + batch_size, len(texts))}/{len(texts)} samples")
      # Convert to numpy arrays for metrics calculation
    predictions = np.array(predictions, dtype=int)
    probabilities = np.array(probabilities, dtype=float)
    true_labels = np.array(true_labels, dtype=int)
    
    # Sanity check - verify label types are consistent
    print(f"Labels type check - predictions: {predictions.dtype}, true_labels: {np.array(true_labels).dtype}")
      # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    balanced_acc = balanced_accuracy_score(true_labels, predictions)  # Better for imbalanced datasets
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='binary')
    
    # Count instances by class
    unique_true_labels, counts_true = np.unique(true_labels, return_counts=True)
    class_distribution = dict(zip(unique_true_labels, counts_true))
    print(f"Class distribution in true labels: {class_distribution}")
    
    # Try to calculate AUC-ROC (may fail if only one class is present)
    try:
        auc_roc = roc_auc_score(true_labels, probabilities)
    except Exception as e:
        auc_roc = float('nan')
        print(f"Warning: Couldn't calculate AUC-ROC score: {e}")
    
    # Create confusion matrix
    cm = confusion_matrix(true_labels, predictions)
      # Print results
    print("\nEvaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    if not np.isnan(auc_roc):
        print(f"AUC-ROC: {auc_roc:.4f}")
    
    print("\nConfusion Matrix:")
    print(cm)
      # Create output directory for visualizations
    output_dir = 'toxigen_evaluation_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Raw counts confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Hate', 'Hate'],
                yticklabels=['Non-Hate', 'Hate'],
                ax=ax1)
    ax1.set_xlabel('Predicted Labels')
    ax1.set_ylabel('True Labels')
    ax1.set_title('Confusion Matrix (Counts)')
    
    # Normalized confusion matrix
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', 
                xticklabels=['Non-Hate', 'Hate'],
                yticklabels=['Non-Hate', 'Hate'],
                ax=ax2)
    ax2.set_xlabel('Predicted Labels')
    ax2.set_ylabel('True Labels')
    ax2.set_title('Confusion Matrix (Normalized)')
    
    plt.suptitle('ToxiGen Model Performance on Russian Annotated Test Data', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/toxigen_russian_confusion_matrix.png', dpi=300)
    print(f"Confusion matrix saved to '{output_dir}/toxigen_russian_confusion_matrix.png'")
    
    # Plot prediction distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(probabilities, bins=50, kde=True)
    plt.axvline(0.5, color='red', linestyle='--', label='Decision Threshold (0.5)')
    plt.xlabel('Hate Probability')
    plt.ylabel('Count')
    plt.title('Distribution of Hate Probabilities')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/hate_probability_distribution.png', dpi=300)
    print(f"Probability distribution saved to '{output_dir}/hate_probability_distribution.png'")
      # Create detailed results dataframe
    results_df = pd.DataFrame({
        'Text': texts,
        'True_Label': true_labels,
        'Predicted_Label': predictions,
        'Hate_Probability': probabilities,
        'Correct_Prediction': true_labels == predictions
    })
    
    # Add confidence and error analysis
    results_df['Confidence'] = np.where(
        probabilities > 0.5, 
        probabilities,
        1 - probabilities
    )
    
    results_df['Error_Type'] = 'Correct'
    # False positives: predicted hate (1) but actually not (0)
    fp_mask = (predictions == 1) & (true_labels == 0)
    results_df.loc[fp_mask, 'Error_Type'] = 'False Positive'
    
    # False negatives: predicted not hate (0) but actually is (1)
    fn_mask = (predictions == 0) & (true_labels == 1)
    results_df.loc[fn_mask, 'Error_Type'] = 'False Negative'
    
    # Export full results
    results_df.to_csv(f'{output_dir}/toxigen_russian_evaluation_results.csv', index=False)
    print(f"Detailed results saved to '{output_dir}/toxigen_russian_evaluation_results.csv'")
    
    # Export most certain and uncertain predictions
    most_certain = results_df.nlargest(20, 'Confidence')
    most_uncertain = results_df.nsmallest(20, 'Confidence')
    
    most_certain.to_csv(f'{output_dir}/most_certain_predictions.csv', index=False)
    most_uncertain.to_csv(f'{output_dir}/most_uncertain_predictions.csv', index=False)
    print(f"Most certain and uncertain predictions saved to '{output_dir}/most_certain_predictions.csv' and '{output_dir}/most_uncertain_predictions.csv'")
    
    # Export error examples
    false_positives = results_df[fp_mask].sort_values(by='Confidence', ascending=False).head(20)
    false_negatives = results_df[fn_mask].sort_values(by='Confidence', ascending=False).head(20)
    
    false_positives.to_csv(f'{output_dir}/false_positive_examples.csv', index=False)
    false_negatives.to_csv(f'{output_dir}/false_negative_examples.csv', index=False)
    print(f"Error examples saved to '{output_dir}/false_positive_examples.csv' and '{output_dir}/false_negative_examples.csv'")
      # Return metrics for further use if needed
    return {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc_roc': auc_roc,
        'confusion_matrix': cm,
        'class_distribution': class_distribution
    }

def setup_gpu():
    """Setup and optimize GPU if available"""
    if torch.cuda.is_available():
        # Set GPU to use mixed precision to speed up computation
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Print GPU info
        gpu_name = torch.cuda.get_device_name(0)
        memory_allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
        memory_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"GPU: {gpu_name}")
        print(f"GPU Memory: {memory_allocated:.2f}GB / {memory_total:.2f}GB")
        return f"GPU: {gpu_name} ({memory_allocated:.2f}GB / {memory_total:.2f}GB)"
    else:
        return "CPU only"

if __name__ == "__main__":
    start_time = time.time()
    print("Starting ToxiGen model evaluation on Russian annotated dataset...")
    
    # Setup GPU if available
    device_info = setup_gpu()
    print(f"Running on {device_info}")
    
    # Run the evaluation
    metrics = evaluate_toxigen_model()
    
    # Save metrics to JSON and CSV for easier analysis
    metrics_df = pd.DataFrame([metrics]).drop('confusion_matrix', axis=1)
    metrics_df.to_json('toxigen_russian_metrics.json', orient='records')
    metrics_df.to_csv('toxigen_russian_metrics.csv', index=False)
    print("Metrics saved to 'toxigen_russian_metrics.json' and 'toxigen_russian_metrics.csv'")
    
    # Print performance summary
    print("\nPerformance Summary:")
    print(f"- ToxiGen correctly identified {metrics['recall']*100:.1f}% of hate content")
    print(f"- Overall accuracy: {metrics['accuracy']*100:.1f}%")
    print(f"- Balanced accuracy: {metrics['balanced_accuracy']*100:.1f}%")
    print(f"- F1 score: {metrics['f1']*100:.1f}%")
    
    elapsed_time = time.time() - start_time
    mins, secs = divmod(elapsed_time, 60)
    time_str = f"{int(mins)}m {secs:.2f}s" if mins > 0 else f"{secs:.2f}s"
    print(f"\nEvaluation complete! Total time: {time_str}")
