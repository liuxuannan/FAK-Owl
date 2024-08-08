from transformers import AutoModelForSequenceClassification
#from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from scipy.special import softmax
import csv
import urllib.request


MODEL = f"../twitter-roberta-base-sentiment"

tokenizer = AutoTokenizer.from_pretrained(MODEL)

# PT  output_hidden_states
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
model.save_pretrained(MODEL)

text = "Good night"
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
scores = output[0][0].detach().numpy()
scores = softmax(scores)
x = 1
d =1
