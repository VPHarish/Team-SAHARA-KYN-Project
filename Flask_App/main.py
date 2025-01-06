from flask import Flask, request, jsonify, render_template
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pytesseract
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import re
import cv2
import pytesseract


import os
import tensorflow as tf
import warnings

warnings.filterwarnings("ignore")

import json
from flask_cors import CORS


from PIL import Image
import requests

from io import BytesIO

os.environ["CUDA_VISIBLE_DEVICES"] = ""

app = Flask(__name__)
cors = CORS(app)

global MODEL
global CLASSES


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


app = Flask(__name__)

# Load your fine-tuned BERT model and tokenizer
model = BertForSequenceClassification.from_pretrained('./bert_model/')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

stop_words = set(stopwords.words('english'))
stop_words.add("rt") # adding rt to remove retweet in dataset

# Removing Emojis
def remove_entity(raw_text):
    entity_regex = r"&[^\s;]+;"
    text = re.sub(entity_regex, "", raw_text)
    return text

# Replacing user tags
def change_user(raw_text):
    regex = r"@([^ ]+)"
    text = re.sub(regex, "", raw_text)
    return text

# Removing URLs
def remove_url(raw_text):
    url_regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    text = re.sub(url_regex, '', raw_text)
    return text

# Removing Unnecessary Symbols
def remove_noise_symbols(raw_text):
    text = raw_text.replace('"', '')
    text = text.replace("'", '')
    text = text.replace("!", '')
    text = text.replace("`", '')
    text = text.replace("..", '')
    text = text.replace(".", '')
    text = text.replace(",", '')
    text = text.replace("#", '')
    text = text.replace(":", '')
    text = text.replace("?", '')
    return text

# Stemming
def stemming(raw_text):
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in raw_text.split()]
    return ' '.join(words)

# Removing stopwords
def remove_stopwords(raw_text):
    tokenize = word_tokenize(raw_text)
    text = [word for word in tokenize if not word.lower() in stop_words]
    text = ' '.join(text)
    return text

def preprocess(data):
    clean = []
    clean = [text.lower() for text in data]
    clean = [change_user(text) for text in clean]
    clean = [remove_entity(text) for text in clean]
    clean = [remove_url(text) for text in clean]
    clean = [remove_noise_symbols(text) for text in clean]
    clean = [stemming(text) for text in clean]
    clean = [remove_stopwords(text) for text in clean]

    return clean

def classify_text(text):

    # Preprocess text
    text_list = [text]
    preprocessed_text = preprocess(text_list)[0]

    # Tokenize the text
    tokenized_input = tokenizer(preprocessed_text, return_tensors='pt')

    # Get model predictions
    output = model(**tokenized_input)
    logits = output.logits

    # Convert logits to probabilities
    probabilities = torch.softmax(logits, dim=1)
    predicted_label = torch.argmax(probabilities, dim=1).item()
    if predicted_label==0:
        predicted_label = "Appropriate"
    else:
        predicted_label = "Inappropriate"

    probability_class_0 = probabilities[0][0].item()
    probability_class_1 = probabilities[0][1].item()

    # Formulate response
    response = {
        "text": text,
        "predicted_class": predicted_label,
        "probabilities": {
            "appropriate": probability_class_0,
            "inappropriate": probability_class_1
        }
    }
    return response


def classify_image(image):
    try:
        image = Image.open(image)
        image = image.resize((128, 128))
        image = np.array(image)
        image = image.astype("float") / 255.0
        image = np.expand_dims(image, axis=0)
        pred = MODEL.predict(image)
        ans =  {"class": CLASSES[int(np.argmax(pred, axis=1))]}
    except:
        ans =  {"Uh oh": "We are down"}

    response = {
        "text": "Image",
        "predicted_class": ans.get("class", "Unknown") if ans.get("class", "Unknown")!="Unknown" else "Pass",
        "probabilities": {
            "appropriate": 0.0 if ans.get("class", "Unknown")!="Unknown" else 1.0,
            "inappropriate": 0.0 if ans.get("class", "Unknown")=="Unknown" else 1.0
        }
    }
    return response


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'text' in request.form:
            text = request.form['text']
            text_response = classify_text(text)
            return jsonify(text_response)
        elif 'image' in request.files:
            image = request.files['image']
            image_response = classify_image(image)
            return jsonify(image_response)
        else:
            return jsonify({"error": "Invalid request"}), 400

    return render_template('index5.html')



if __name__ == '__main__':
    MODEL_PATH = os.path.abspath("./Image_Moderator/models/image/dump/mobile_net.h5")
    MODEL = tf.keras.models.load_model(MODEL_PATH)
    CLASSES = ["control", "gore", "pornography"]
    app.run(threaded=True, debug=True)
