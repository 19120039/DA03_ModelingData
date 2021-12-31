# Import Thu vien
import streamlit as st
import pandas as pd
import pickle
import numpy as np
import pandas as pd 
import re
import time
from underthesea import sent_tokenize
from underthesea import classify
from underthesea import word_tokenize
import seaborn as sns 
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from PIL import Image


tree = pickle.load(open("decisiontree.sav", "rb"))


clf = pickle.load(open("logisticregression.sav", "rb"))


vector = pickle.load(open("vectorize.sav", "rb"))


# Các hàm tiền xử lý ================================

file_stopwords = "stopwords.txt"

with open(file_stopwords, 'r', encoding="utf-8") as f:
    stopwords = f.readlines()
    stop_set = set(m.strip() for m in stopwords)
    stopwords =  list(frozenset(stop_set))


def remove_trailing_newline(x):
    return x.replace("\n", ". ")

def sentencing(text):
    return sent_tokenize(text)

def remove_stopwords(word_list):
    for el in word_list:
        if len(el)==0:
            word_list.remove(el)
            continue
        for sw in stopwords:
            if sw in el:
                el.replace(sw, "")
    return word_list

def combine(list_sentence):
    temp = []
    for i in list_sentence:
        temp.append(i)
    return str(temp)

def Precprocess_input(user_input):
    #user_input = np.asarray(user_input)
    df = pd.DataFrame({"text": [user_input]})
    df['text'] = df['text'].apply(remove_trailing_newline)
    df['text'] = df['text'].apply(sentencing)
    df['text'] = df['text'].apply(remove_stopwords)
    df['text'] = df['text'].apply(combine)
    df_vectorizer = vector.transform(df["text"])
    return df_vectorizer

#=================================================================



st.set_page_config(page_title="Awesome Tool-App that detect any Fake News!", page_icon="🐞", layout="centered")

st.title("Awesome Tool-App that detect any Fake News!")

st.sidebar.write(
    "This Tool-App was built by:\n\n\n\n\n"
    "- Phan Minh Triet (19120039) \n\n\n\n" 
    "- Tran Duc Thuy (19120138) \n\n\n\n"
    "- Nguyen Thanh Hien (19120503) \n\n\n\n"
    "From class 19_21, NMKHDL\n\n\n\n"
    "For details, please contact via trietphanminh@gmail.com"
)

form = st.form(key="annotation")

with form:
    #cols = st.columns((10, 10))
    model_type = st.selectbox("Choose Model:", ["Decicison Tree", "Logistic Regression"], index = 1)
    if (model_type == "Decicison Tree"):
       image = Image.open('decisiontree.jpg')
       st.image(image)
    else:
       image = Image.open('fake-news-3.png')
       st.image(image)
    user_input = st.text_input("Write your incredible News here:")
    if (user_input == ""):
        st.error("Oops, You forgot to write your News!")
        submitted = False
    cols = st.columns(2)
    submitted = st.form_submit_button(label = "Enter your News")
    if (user_input == ""):
        submitted = 99

if submitted != 99:
    user_input_df = Precprocess_input(user_input)
    with st.spinner('Loading the result...'):
            time.sleep(5)
            st.write('Done!')
    if (model_type == "Decicison Tree"):
        output = tree.predict(user_input_df)
        if(output == 0):
            st.warning("Becareful, this is Fake News!!! \n\n\n If not, Please check your News!")
        if (output == 1):
            st.success("Congratulation, this is Real News!!!")
    if (model_type == "Logistic Regression"):
        output = clf.predict(user_input_df)
        if(output == 0):
            st.warning("Becareful, this is Fake News!!! \n\n\n If not, Please check your News!")
        if (output == 1):
            st.success("Congratulation, this is Real News!!!")
            st.balloons()
    

    
