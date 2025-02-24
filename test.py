from transformers import pipeline
from checklist.pred_wrapper import PredictorWrapper
from checklist.test_types import MFT, INV, DIR
from checklist.test_suite import TestSuite
from checklist.expect import Expect
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Sample dataset creation (replace this with your actual dataset)
def create_sample_dataset():
    data = {
        'text': [
            "This product is amazing!",
            "Terrible service, would not recommend",
            "I love this app, very useful",
            "Waste of money, don't buy",
            "Great experience with customer support",
            # Add more examples...
        ],
        'label': [1, 0, 1, 0, 1]  # 1 for positive, 0 for negative
    }
    return pd.DataFrame(data)

# Initialize BERT pipeline for binary classification
classifier = pipeline(
    task="sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

# Wrapper class for binary classification
class BinaryClassificationWrapper(PredictorWrapper):
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def __call__(self, texts):
        # Process batch of texts
        results = self.pipeline(texts)
        
        # Convert to binary classification probabilities
        # Output shape: (n_samples, 2) for [negative_prob, positive_prob]
        probs = np.zeros((len(texts), 2))
        
        for i, result in enumerate(results):
            if result['label'] == 'POSITIVE':
                probs[i] = [1 - result['score'], result['score']]
            else:
                probs[i] = [result['score'], 1 - result['score']]
                
        return probs

# Test suite setup
def create_test_suite(dataset):
    suite = TestSuite()
    model = BinaryClassificationWrapper(classifier)

    # 2. Invariance Tests
    def change_irrelevant_details(text):
        """Change irrelevant details that shouldn't affect sentiment"""
        variations = [
            text.replace('!', '.'),
            text.replace('I', 'We'),
            text + ' Actually',
        ]
        return variations[np.random.randint(0, len(variations))]

    test = INV(
        name="irrelevant_changes",
        data=dataset['text'].tolist()[:50],  # Use subset of real examples
        perturb=change_irrelevant_details
    )
    suite.add(test)

    # 3. Directional Tests using real examples
    def add_emphasis(text):
        """Add emphasis to strengthen the sentiment"""
        return f"Definitely {text.lower()}"

    # Use subset of clear positive/negative examples
    strong_examples = dataset[abs(dataset['label'] - 0.5) > 0.4]['text'].tolist()[:30]
    
    test = DIR(
        name="emphasis_effect",
        data=strong_examples,
        perturb=add_emphasis,
        expect=Expect.increase
    )
    suite.add(test)

    return suite, model

# Function to analyze results
def analyze_test_results(results, dataset):
    analysis = {
        'summary': {
            'total_tests': len(results.tests),
            'passed': sum(1 for test in results.tests if test.passed),
            'failed': sum(1 for test in results.tests if not test.passed),
            'dataset_size': len(dataset),
            'class_distribution': dataset['label'].value_counts().to_dict()
        },
        'test_details': []
    }

    for test in results.tests:
        test_details = {
            'name': test.name,
            'type': test.type,
            'success_rate': f"{(test.success_rate * 100):.2f}%",
            'status': 'PASS' if test.passed else 'FAIL',
            'n_samples': len(test.results),
            'failure_examples': []
        }
        
        # Add examples of failures if any
        if not test.passed:
            for failure in test.failures[:5]:  # Show first 5 failures
                test_details['failure_examples'].append({
                    'text': failure.data,
                    'expected': failure.expected,
                    'got': failure.pred
                })
                
        analysis['test_details'].append(test_details)
    
    return analysis

# Main execution function
def run_binary_classification_tests(dataset=None):
    if dataset is None:
        dataset = create_sample_dataset()
    
    # Split dataset into train/test if needed
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
    
    # Create and run test suite
    suite, model = create_test_suite(train_data)
    results = suite.run(model)
    
    # Analyze results
    analysis = analyze_test_results(results, train_data)
    
    return analysis

# Example usage
if __name__ == "__main__":
    # You can replace this with your own dataset
    dataset = create_sample_dataset()
    analysis = run_binary_classification_tests(dataset)
    
    print("\nTest Results Summary:")
    print(f"Total Tests: {analysis['summary']['total_tests']}")
    print(f"Passed: {analysis['summary']['passed']}")
    print(f"Failed: {analysis['summary']['failed']}")
    print("\nClass Distribution:")
    for label, count in analysis['summary']['class_distribution'].items():
        print(f"Class {label}: {count} samples")