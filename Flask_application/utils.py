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


from tensorflow.keras.models import load_model

def clean_up_sentence(sentence):
    lemmatizer = WordNetLemmatizer()
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    words = pickle.load(open('Flask_application/words.pkl','rb'))
    sentence_word = clean_up_sentence(sentence)
    bag = [0] * len(words)
    
    for w in sentence_word:
        for i, word in enumerate(words):
            if word == w:
              bag[i] = 1  
    
    return np.array(bag)

def predict_class(sentence):
    classes = pickle.load(open('Flask_application/classes.pkl','rb'))
    model = load_model('Flask_application/chatbot_model.keras')
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    EROR_THRESH = 0.25
    
    results = [[i,r] for i,r in enumerate(res) if r > EROR_THRESH]
    results.sort(key = lambda x: x[1], reverse=True)
    
    return_list = []
    
    for r in results:
        return_list.append({'intent' : classes[r[0]], 'probability': str(r[1])})
        
    return return_list


def get_response(intents_list):
    with open('Model_training/chatbot_intents.json', 'r') as f:
        intents_json = json.load(f)
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    
    for i in list_of_intents:
        if i['tag']== tag:
            result = random.choice(i['responses'])
            break
    
    return result
    
    



    
    
    
    