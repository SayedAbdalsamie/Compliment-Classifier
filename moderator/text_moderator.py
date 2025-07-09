"""
Text moderation system with ensemble prediction and majority voting.
"""
import os
import joblib
from typing import Dict, Tuple, Any
from text_preprocessor import TextPreprocessor
from config import MODELS_DIR, MAJORITY_THRESHOLD, DATASETS


class TextModerator:
    """Main text moderation system with ensemble prediction."""
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.models = {}
        self.vectorizers = {}
        self.dataset_names = []
        self.load_all_models()
    
    def load_all_models(self):
        """Load all trained models and vectorizers."""
        print("Loading trained models...")
        
        # Get list of available models
        if not os.path.exists(MODELS_DIR):
            raise FileNotFoundError(f"Models directory not found: {MODELS_DIR}")
        
        model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('_model.joblib')]
        
        if not model_files:
            raise FileNotFoundError("No trained models found in models directory")
        
        # Load each model and its corresponding vectorizer
        for model_file in model_files:
            dataset_name = model_file.replace('_model.joblib', '')
            vectorizer_file = f'{dataset_name}_vectorizer.joblib'
            
            model_path = os.path.join(MODELS_DIR, model_file)
            vectorizer_path = os.path.join(MODELS_DIR, vectorizer_file)
            
            if os.path.exists(vectorizer_path):
                try:
                    model = joblib.load(model_path)
                    vectorizer = joblib.load(vectorizer_path)
                    
                    self.models[dataset_name] = model
                    self.vectorizers[dataset_name] = vectorizer
                    self.dataset_names.append(dataset_name)
                    
                    print(f"Loaded model: {dataset_name}")
                except Exception as e:
                    print(f"Error loading model {dataset_name}: {e}")
            else:
                print(f"Vectorizer not found for {dataset_name}")
        
        if not self.models:
            raise RuntimeError("No models loaded successfully")
        
        print(f"Successfully loaded {len(self.models)} models: {', '.join(self.dataset_names)}")
    
    def predict_single_model(self, text: str, dataset_name: str) -> Tuple[int, float]:
        """
        Get prediction from a single model.
        
        Args:
            text: Input text to classify
            dataset_name: Name of the dataset/model to use
            
        Returns:
            Tuple of (prediction, confidence)
        """
        if dataset_name not in self.models:
            raise ValueError(f"Model {dataset_name} not found")
        
        # Preprocess text
        cleaned_text = self.preprocessor.clean_text(text)
        
        if not cleaned_text.strip():
            # Empty text after cleaning - default to Accept
            return 1, 0.5
        
        # Vectorize text
        vectorizer = self.vectorizers[dataset_name]
        text_tfidf = vectorizer.transform([cleaned_text])
        
        # Get prediction and probability
        model = self.models[dataset_name]
        prediction = model.predict(text_tfidf)[0]
        probabilities = model.predict_proba(text_tfidf)[0]
        confidence = max(probabilities)
        
        return int(prediction), float(confidence)
    
    def predict_all_models(self, text: str) -> Dict[str, Dict[str, Any]]:
        """
        Get predictions from all models.
        
        Args:
            text: Input text to classify
            
        Returns:
            Dictionary with predictions from each model
        """
        predictions = {}
        
        for dataset_name in self.dataset_names:
            try:
                prediction, confidence = self.predict_single_model(text, dataset_name)
                
                predictions[dataset_name] = {
                    'prediction': prediction,
                    'decision': 'Accept' if prediction == 1 else 'Reject',
                    'confidence': confidence,
                    'model_name': DATASETS.get(dataset_name, {}).get('name', dataset_name)
                }
            except Exception as e:
                print(f"Error predicting with {dataset_name}: {e}")
                predictions[dataset_name] = {
                    'prediction': 1,  # Default to Accept on error
                    'decision': 'Accept',
                    'confidence': 0.0,
                    'error': str(e),
                    'model_name': DATASETS.get(dataset_name, {}).get('name', dataset_name)
                }
        
        return predictions
    
    def majority_vote(self, predictions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Apply majority voting to combine predictions.
        
        Args:
            predictions: Dictionary with predictions from each model
            
        Returns:
            Final decision with voting details
        """
        # Count votes
        reject_votes = 0
        accept_votes = 0
        total_confidence = 0
        valid_predictions = 0
        
        for dataset_name, pred_info in predictions.items():
            if 'error' not in pred_info:
                if pred_info['prediction'] == 0:  # Reject
                    reject_votes += 1
                else:  # Accept
                    accept_votes += 1
                
                total_confidence += pred_info['confidence']
                valid_predictions += 1
        
        # Apply majority voting rule
        final_decision = 'Reject' if reject_votes >= MAJORITY_THRESHOLD else 'Accept'
        avg_confidence = total_confidence / valid_predictions if valid_predictions > 0 else 0.0
        
        return {
            'final_decision': final_decision,
            'reject_votes': reject_votes,
            'accept_votes': accept_votes,
            'total_models': len(predictions),
            'valid_predictions': valid_predictions,
            'average_confidence': avg_confidence,
            'voting_threshold': MAJORITY_THRESHOLD,
            'voting_rule': f'Reject if {MAJORITY_THRESHOLD}+ models say Reject, else Accept'
        }
    
    def moderate_text(self, text: str) -> Dict[str, Any]:
        """
        Complete text moderation with all models and majority voting.
        
        Args:
            text: Input text to moderate
            
        Returns:
            Complete moderation result
        """
        if not isinstance(text, str):
            text = str(text)
        
        # Get predictions from all models
        model_predictions = self.predict_all_models(text)
        
        # Apply majority voting
        voting_result = self.majority_vote(model_predictions)
        
        # Combine results
        result = {
            'input_text': text,
            'preprocessed_text': self.preprocessor.clean_text(text),
            'model_predictions': model_predictions,
            'voting_result': voting_result,
            'final_decision': voting_result['final_decision']
        }
        
        return result
    
    def print_moderation_result(self, result: Dict[str, Any]):
        """Print formatted moderation result."""
        print("\n" + "="*60)
        print("TEXT MODERATION RESULT")
        print("="*60)
        print(f"Input Text: {result['input_text']}")
        print(f"Preprocessed: {result['preprocessed_text']}")
        print("\nIndividual Model Predictions:")
        print("-" * 40)
        
        for dataset_name, pred_info in result['model_predictions'].items():
            model_name = pred_info.get('model_name', dataset_name)
            decision = pred_info['decision']
            confidence = pred_info['confidence']
            
            if 'error' in pred_info:
                print(f"{model_name:20}: ERROR - {pred_info['error']}")
            else:
                print(f"{model_name:20}: {decision:6} (confidence: {confidence:.3f})")
        
        print("\nVoting Summary:")
        print("-" * 40)
        voting = result['voting_result']
        print(f"Reject votes: {voting['reject_votes']}")
        print(f"Accept votes: {voting['accept_votes']}")
        print(f"Voting rule: {voting['voting_rule']}")
        print(f"Average confidence: {voting['average_confidence']:.3f}")
        
        print(f"\nFINAL DECISION: {result['final_decision']}")
        print("="*60)


if __name__ == "__main__":
    # Test the moderator
    moderator = TextModerator()
    
    test_texts = [
        "This is a normal, clean message.",
        "You are such an idiot and I hate you!",
        "Click here to win $1000000 now!!!",
        "I disagree with your opinion, but I respect it.",
        "F*** you and your stupid ideas!"
    ]
    
    for text in test_texts:
        result = moderator.moderate_text(text)
        moderator.print_moderation_result(result)
