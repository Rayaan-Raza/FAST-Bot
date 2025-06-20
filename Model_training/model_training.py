import os
import random
import pickle
import numpy as np
import json
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  #Suppress TensorFlow warnings 

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('chatbot_intents.json'))

words = []
classes = []
documents = []

ignore_letters = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
            
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters] 
 