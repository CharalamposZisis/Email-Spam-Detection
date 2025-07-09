import streamlit as st
import pickle 
import string 
from nltk.corpus import stopwords
import nltk
nltk.download('punkt')
from nltk.stem import PorterStemmer
import re

ps = PorterStemmer()

def transform_text(text):
    # Μετατροπή σε μικρά γράμματα
    text = text.lower()
    
    # Αφαίρεση σημείων στίξης και διαχωρισμός σε λέξεις (tokens)
    words = re.findall(r'\b\w+\b', text)
    
    # Αφαίρεση stopwords (χρησιμοποιούμε built-in stopwords του sklearn)
    words = [word for word in words if word not in ENGLISH_STOP_WORDS]
    
    # Επιστροφή του κειμένου σαν ενιαίο string
    return " ".join(words)


tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email/SMS News Detection")

input_sms = st.text_input("Enter the message:")

#Data Preprocessing
transformed_sms = transform_text(input_sms)

#Vectorize the input
vector_input = tfidf.transform([transformed_sms])

#Predict
result = model.predict(vector_input)[0]

#Display the result
if result == 1:
    st.header("The message is Fake.")
else:
    st.header("The message is not Fake.")