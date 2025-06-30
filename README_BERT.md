# FAST Bot - BERT-Based Chatbot

This project implements a BERT-based chatbot for FAST University using fine-tuned BERT instead of traditional bag-of-words approach.

## Features

- **BERT Fine-tuning**: Uses BERT (Bidirectional Encoder Representations from Transformers) for better understanding of context and semantics
- **Improved Accuracy**: Better performance compared to bag-of-words approach
- **Context Awareness**: BERT understands word relationships and context better
- **Fallback Support**: Includes fallback to original bag-of-words method if BERT fails
- **Flask Web Interface**: Web-based chatbot interface

## Project Structure

```
FAST-bot/
├── Model_training/
│   ├── bert_model_training.py      # BERT training script
│   ├── bert_prediction.py          # BERT prediction script
│   ├── requirements_bert.txt       # BERT training dependencies
│   └── chatbot_intents.json        # Training data
├── Flask_application/
│   ├── app.py                      # Flask web app
│   ├── utils_bert.py               # BERT-based utilities
│   ├── requirements_bert.txt       # Flask BERT dependencies
│   └── templates/
│       └── index.html              # Web interface
└── model/                          # Saved models (created after training)
```

## Installation

### 1. Install Dependencies

For training:
```bash
cd Model_training
pip install -r requirements_bert.txt
```

For Flask application:
```bash
cd Flask_application
pip install -r requirements_bert.txt
```

### 2. Download NLTK Data (if using fallback)
```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

## Training the BERT Model

### 1. Navigate to Training Directory
```bash
cd Model_training
```

### 2. Run BERT Training
```bash
python bert_model_training.py
```

This will:
- Load your `chatbot_intents.json` data
- Fine-tune a BERT model for intent classification
- Save the trained model to `model/` directory
- Save tokenizer, label encoder, and label mapping

### Training Parameters

You can modify these parameters in `bert_model_training.py`:

- **Learning Rate**: `lr=2e-5` (default)
- **Batch Size**: `batch_size=16` (default)
- **Epochs**: `num_epochs=3` (default)
- **Max Sequence Length**: `max_len=128` (default)

### Expected Output Files

After training, you'll have:
- `model/bert_chatbot_model.pth` - Model weights
- `model/label_encoder.pkl` - Label encoder
- `model/label_mapping.pkl` - Label mapping
- `model/config.json` - BERT configuration
- `model/vocab.txt` - BERT vocabulary
- `model/tokenizer_config.json` - Tokenizer configuration

## Testing the BERT Model

### 1. Test Prediction Script
```bash
cd Model_training
python bert_prediction.py
```

This will test the model with sample questions and show:
- Predicted intent
- Confidence score
- Generated response

### 2. Interactive Testing
```python
from bert_prediction import BERTChatbotPredictor

predictor = BERTChatbotPredictor()
response = predictor.get_response("What is the admission process?")
print(response)
```

## Running the Flask Web Application

### 1. Navigate to Flask Directory
```bash
cd Flask_application
```

### 2. Update Utils Import (if needed)
If you want to use BERT instead of bag-of-words, update `app.py`:

```python
# Change this line:
from utils import get_response, predict_class

# To this:
from utils_bert import get_response, predict_class
```

### 3. Run Flask App
```bash
python app.py
```

### 4. Access Web Interface
Open your browser and go to: `http://localhost:5000`

## Configuration

### Model Path
The BERT model looks for files in the `model/` directory. Make sure the path is correct:

```python
# In utils_bert.py
model_path='../model/'  # Relative to Flask_application/
```

### Confidence Threshold
Adjust the confidence threshold for predictions:

```python
# In utils_bert.py
error_threshold=0.25  # Default threshold
```

### Device Selection
The model automatically uses GPU if available, otherwise CPU:

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

## Performance Comparison

### BERT Advantages
- **Better Context Understanding**: Understands word relationships
- **Higher Accuracy**: Generally better performance on intent classification
- **Semantic Understanding**: Better at understanding synonyms and paraphrases
- **Transfer Learning**: Leverages pre-trained language knowledge

### Bag-of-Words Advantages
- **Faster Inference**: Lighter computational requirements
- **Smaller Model Size**: Less storage space
- **Simplicity**: Easier to understand and debug

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in training
   - Use CPU instead of GPU
   - Reduce max sequence length

2. **Model Loading Errors**
   - Ensure all model files are in the correct directory
   - Check file paths in the code
   - Verify model was trained successfully

3. **Import Errors**
   - Install all required dependencies
   - Check Python version compatibility
   - Ensure transformers and torch versions are compatible

### Fallback Mode

If BERT fails to load, the system automatically falls back to the original bag-of-words method. Check the console for error messages.

## Customization

### Adding New Intents
1. Add new intent to `chatbot_intents.json`
2. Retrain the BERT model
3. Restart the Flask application

### Modifying Responses
Edit the responses in `chatbot_intents.json` and restart the application.

### Changing BERT Model
You can use different BERT variants by changing the model name:

```python
# In bert_model_training.py
model_name = 'bert-base-uncased'  # Default
# Other options:
# model_name = 'bert-large-uncased'
# model_name = 'distilbert-base-uncased'
```

## Requirements

- Python 3.7+
- PyTorch 1.9+
- Transformers 4.20+
- Flask 2.0+
- CUDA (optional, for GPU acceleration)

## License

This project is for educational purposes. Please ensure you comply with BERT's license terms from Hugging Face. 