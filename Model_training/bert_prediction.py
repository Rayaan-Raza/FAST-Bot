import torch
import pickle
import json
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np

class BERTChatbotPredictor:
    def __init__(self, model_path='model/', device=None):
        """
        Initialize BERT chatbot predictor
        
        Args:
            model_path: Path to the saved model directory
            device: Device to run inference on ('cuda' or 'cpu')
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        self.model_path = model_path
        
        # Load tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        
        # Load label encoder and mapping
        with open(f'{model_path}/label_encoder.pkl', 'rb') as f:
            self.label_encoder = pickle.load(f)
            
        with open(f'{model_path}/label_mapping.pkl', 'rb') as f:
            self.label_mapping = pickle.load(f)
        
        # Load model
        self.model = BertForSequenceClassification.from_pretrained(
            model_path,
            num_labels=len(self.label_encoder.classes_)
        )
        self.model.load_state_dict(torch.load(f'{model_path}/bert_chatbot_model.pth', map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Load intents for responses
        with open('chatbot_intents.json', 'r', encoding='utf-8') as f:
            self.intents = json.load(f)
        
        # Create intent to responses mapping
        self.intent_responses = {}
        for intent in self.intents['intents']:
            self.intent_responses[intent['tag']] = intent['responses']
    
    def predict_intent(self, text, max_length=128):
        """
        Predict the intent of a given text
        
        Args:
            text: Input text to classify
            max_length: Maximum sequence length
            
        Returns:
            tuple: (predicted_intent, confidence_score)
        """
        # Tokenize input
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            
            # Get predicted class and confidence
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
            
            # Convert back to original label
            predicted_intent = self.label_mapping[predicted_class]
            
        return predicted_intent, confidence
    
    def get_response(self, text, confidence_threshold=0.5):
        """
        Get a response for the given text
        
        Args:
            text: Input text
            confidence_threshold: Minimum confidence to return a response
            
        Returns:
            str: Response text
        """
        predicted_intent, confidence = self.predict_intent(text)
        
        if confidence < confidence_threshold:
            return "I'm sorry, I'm not sure how to respond to that. Could you rephrase your question?"
        
        # Get random response from the predicted intent
        responses = self.intent_responses.get(predicted_intent, ["I'm not sure how to respond to that."])
        return np.random.choice(responses)
    
    def get_intent_with_confidence(self, text):
        """
        Get intent prediction with confidence score
        
        Args:
            text: Input text
            
        Returns:
            dict: Contains 'intent', 'confidence', and 'responses'
        """
        predicted_intent, confidence = self.predict_intent(text)
        
        return {
            'intent': predicted_intent,
            'confidence': confidence,
            'responses': self.intent_responses.get(predicted_intent, [])
        }

def main():
    """Example usage of the BERT chatbot predictor"""
    # Initialize predictor
    predictor = BERTChatbotPredictor()
    
    # Test examples
    test_questions = [
        "Hello, how are you?",
        "What is the admission process?",
        "Where are FAST campuses located?",
        "What documents do I need for admission?",
        "How can I contact FAST?",
        "What is the grading system?",
        "This is a completely random question about cooking"
    ]
    
    print("BERT Chatbot Predictor - Test Results")
    print("=" * 50)
    
    for question in test_questions:
        print(f"\nQuestion: {question}")
        
        # Get detailed prediction
        result = predictor.get_intent_with_confidence(question)
        print(f"Predicted Intent: {result['intent']}")
        print(f"Confidence: {result['confidence']:.4f}")
        
        # Get response
        response = predictor.get_response(question)
        print(f"Response: {response}")
        print("-" * 30)

if __name__ == '__main__':
    main() 