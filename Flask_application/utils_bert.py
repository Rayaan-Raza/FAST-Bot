import os
import random
import pickle
import numpy as np
import json
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class EnhancedBERTClassifier(nn.Module):
    def __init__(self, bert_model, num_labels, dropout_rate=0.3):
        super().__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(bert_model.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        # Get the base BERT outputs (without the classifier)
        bert_outputs = self.bert.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Use the [CLS] token from last hidden state
        pooled_output = bert_outputs.last_hidden_state[:, 0, :]  # Shape: (batch_size, 768)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)  # Shape: (batch_size, num_labels)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.classifier.out_features), labels.view(-1))
        
        return type('Outputs', (), {'loss': loss, 'logits': logits})()

class BERTPredictor:
    def __init__(self, model_path=None, device=None):
        """
        Initialize BERT predictor for Flask app
        
        Args:
            model_path: Path to the saved model directory (if None, will use relative path from script location)
            device: Device to run inference on ('cuda' or 'cpu')
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        # Use absolute model directory and model file
        self.model_dir = r'C:\Users\rayaan\Desktop\FAST-bot\model'
        self.model_file = os.path.join(self.model_dir, 'bert_chatbot_model_best.pth')
        
        # Load tokenizer from model directory
        self.tokenizer = BertTokenizer.from_pretrained(self.model_dir)
        
        # Load label encoder and mapping
        with open(os.path.join(self.model_dir, 'label_encoder.pkl'), 'rb') as f:
            self.label_encoder = pickle.load(f)
        with open(os.path.join(self.model_dir, 'label_mapping.pkl'), 'rb') as f:
            self.label_mapping = pickle.load(f)
        
        # Instantiate model with the same architecture as training
        num_labels = len(self.label_encoder.classes_)
        base_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
        self.model = EnhancedBERTClassifier(base_model, num_labels, dropout_rate=0.3)
        
        # Load the trained weights
        checkpoint = torch.load(self.model_file, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best model with accuracy: {checkpoint.get('best_accuracy', 'unknown'):.4f}")
        else:
            self.model.load_state_dict(checkpoint)
            print("Loaded model (state dict only)")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Load intents for responses
        script_dir = os.path.dirname(os.path.abspath(__file__))
        intents_path = os.path.join(script_dir, '..', 'Model_training', 'chatbot_intents.json')
        with open(intents_path, 'r', encoding='utf-8') as f:
            self.intents = json.load(f)
        
        # Create intent to responses mapping
        self.intent_responses = {}
        for intent in self.intents['intents']:
            self.intent_responses[intent['tag']] = intent['responses']
    
    def predict_class(self, sentence, max_length=128, error_threshold=0.25):
        """
        Predict the intent class of a sentence
        
        Args:
            sentence: Input sentence
            max_length: Maximum sequence length
            error_threshold: Minimum confidence threshold
            
        Returns:
            list: List of dictionaries with intent and probability
        """
        # Tokenize input
        encoding = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        token_type_ids = encoding['token_type_ids'].to(self.device)
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                token_type_ids=token_type_ids
            )
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            
            # Get all probabilities above threshold
            probs = probabilities[0].cpu().numpy()
            results = [[i, prob] for i, prob in enumerate(probs) if prob > error_threshold]
            results.sort(key=lambda x: x[1], reverse=True)
            
            return_list = []
            for r in results:
                intent = self.label_mapping[r[0]]
                return_list.append({'intent': intent, 'probability': str(r[1])})
            
            return return_list
    
    def get_response(self, intents_list):
        """
        Get a response based on predicted intents
        
        Args:
            intents_list: List of predicted intents with probabilities
            
        Returns:
            str: Response text
        """
        if not intents_list:
            return "I'm sorry, I'm not sure how to respond to that. Could you rephrase your question?"
        
        tag = intents_list[0]['intent']
        responses = self.intent_responses.get(tag, ["I'm not sure how to respond to that."])
        
        return random.choice(responses)

# Initialize global predictor instance
try:
    bert_predictor = BERTPredictor()
    print("BERT model loaded successfully!")
    MODEL_TYPE = "BERT"
except Exception as e:
    print(f"Error loading BERT model: {e}")
    bert_predictor = None
    MODEL_TYPE = "Bag-of-Words (Fallback)"

def get_model_type():
    """
    Get the current model type being used
    
    Returns:
        str: "BERT" or "Bag-of-Words (Fallback)"
    """
    return MODEL_TYPE

def predict_class(sentence):
    """
    Wrapper function to maintain compatibility with existing Flask app
    
    Args:
        sentence: Input sentence
        
    Returns:
        list: List of dictionaries with intent and probability
    """
    if bert_predictor is None:
        # Fallback to original bag-of-words method if BERT fails
        return fallback_predict_class(sentence)
    
    return bert_predictor.predict_class(sentence)

def get_response(intents_list):
    """
    Wrapper function to maintain compatibility with existing Flask app
    
    Args:
        intents_list: List of predicted intents with probabilities
        
    Returns:
        str: Response text
    """
    if bert_predictor is None:
        # Fallback to original method if BERT fails
        return fallback_get_response(intents_list)
    
    return bert_predictor.get_response(intents_list)

# Fallback functions using original bag-of-words method
def fallback_predict_class(sentence):
    """Fallback to original bag-of-words prediction method"""
    try:
        import nltk
        from nltk.stem import WordNetLemmatizer
        from tensorflow.keras.models import load_model
        
        # Get script directory for absolute paths
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        def clean_up_sentence(sentence):
            lemmatizer = WordNetLemmatizer()
            sentence_words = nltk.word_tokenize(sentence)
            sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
            return sentence_words

        def bag_of_words(sentence):
            words_path = os.path.join(script_dir, 'words.pkl')
            words = pickle.load(open(words_path,'rb'))
            sentence_word = clean_up_sentence(sentence)
            bag = [0] * len(words)
            
            for w in sentence_word:
                for i, word in enumerate(words):
                    if word == w:
                      bag[i] = 1  
            
            return np.array(bag)
        
        classes_path = os.path.join(script_dir, 'classes.pkl')
        model_path = os.path.join(script_dir, 'chatbot_model.keras')
        
        # Check if files exist
        if not os.path.exists(classes_path):
            print(f"Classes file not found: {classes_path}")
            return []
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            return []
        if not os.path.exists(os.path.join(script_dir, 'words.pkl')):
            print(f"Words file not found: {os.path.join(script_dir, 'words.pkl')}")
            return []
        
        classes = pickle.load(open(classes_path,'rb'))
        
        # Try to load the model with custom_objects to handle any custom layers
        try:
            model = load_model(model_path, compile=False)
        except Exception as e:
            print(f"Failed to load Keras model: {e}")
            return []
        
        bow = bag_of_words(sentence)
        res = model.predict(np.array([bow]), verbose=0)[0]
        EROR_THRESH = 0.25
        
        results = [[i,r] for i,r in enumerate(res) if r > EROR_THRESH]
        results.sort(key = lambda x: x[1], reverse=True)
        
        return_list = []
        
        for r in results:
            return_list.append({'intent' : classes[r[0]], 'probability': str(r[1])})
            
        return return_list
        
    except Exception as e:
        print(f"Fallback prediction failed: {e}")
        return []

def fallback_get_response(intents_list):
    """Fallback to original response method"""
    try:
        # Get script directory for absolute paths
        script_dir = os.path.dirname(os.path.abspath(__file__))
        intents_path = os.path.join(script_dir, '..', 'Model_training', 'chatbot_intents.json')
        
        with open(intents_path, 'r') as f:
            intents_json = json.load(f)
        
        if not intents_list:
            return "I'm sorry, I'm not sure how to respond to that."
            
        tag = intents_list[0]['intent']
        list_of_intents = intents_json['intents']
        
        for i in list_of_intents:
            if i['tag']== tag:
                result = random.choice(i['responses'])
                break
        else:
            result = "I'm not sure how to respond to that."
        
        return result
        
    except Exception as e:
        print(f"Fallback response failed: {e}")
        return "I'm sorry, I'm having trouble processing your request." 