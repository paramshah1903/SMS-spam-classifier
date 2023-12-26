import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import os

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Determine the directory of the script
script_directory = os.path.dirname(os.path.realpath(__file__))

# Load the TfidfVectorizer
tfidf_path = os.path.join(script_directory, 'vectorizer.pkl')
tfidf = pickle.load(open(tfidf_path, 'rb'))

# Load the model
model_path = os.path.join(script_directory, 'model.pkl')
model = pickle.load(open(model_path, 'rb'))

st.title("SMS-Spam Classifier")

input_sms = st.text_area("Enter the message:")

if st.button('Predict'):
    transformed_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)[0]

    if result == 1:
        st.header("Spam")
    else:
        st.header("Not spam")
