#!/usr/bin/env python
# coding: utf-8

# In[10]:


import requests
import datetime

import os
import re
import string
import copy
import math
import glob
import pickle

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline

import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from collections import Counter

from num2words import num2words

from bs4 import BeautifulSoup


# In[26]:


MODEL_FOLDER = "model"

tfidf_transformer_model = "tfidf_transformer"
count_vectorizer_model = "count_vectorizer"

tfidf_transformer_version = "v4"
count_vectorizer_version = "v4"

MAX_DF = 0.95
MIN_DF = 0.25

INPUT_FOLDER = "prediction"
OUTPUT_FOLDER = "prediction"
OUTPUT_FILE = "prediction_text.csv"
MODEL_FOLDER = "model"

PROCESSED_DATA_FOLDER = "data/4_all_data_preprocessed"
ISW_DATA_FILE = "all_isw.csv"
WEATHER_EVENTS_DATA_FILE = "features_generated_v1.csv"

OUTPUT_DATA_FILE = "all_features"

tfidf_transformer_model = "tfidf_transformer"
count_vectorizer_model = "count_vectorizer"

tfidf_transformer_version = "v4"
count_vectorizer_version = "v4"

files_by_days = glob.glob(f"{INPUT_FOLDER}/*.html")


# In[12]:


def save_page(url, file_name):

    # Send a GET request to the URL and retrieve the response
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Save the response content to a file
        with open(f"{file_name}.html", "wb") as f:
            f.write(response.content)
        print("Page downloaded successfully.")
        return True
    else:
        print(f"Error downloading page. Status code: {response.status_code}")
        return False


# In[ ]:





# In[13]:


# Get the current date
today_date = datetime.date.today()
week_ago = today_date - datetime.timedelta(days=7)
# Define a list of month names
month_names = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december']

# Iterate over the dates from the start date to the end date
suc_downl_flag = False
while suc_downl_flag != True:
    month_name = month_names[today_date.month - 1]
    day = int(today_date.strftime("%d"))
    name = today_date.strftime("%Y-%m-%d")
    year = today_date.strftime("%Y")
    url = ("https://www.understandingwar.org/backgrounder/russian-offensive-campaign-assessment-%s-%d-%s"%(month_name, day, year))
    print(url)
    suc_downl_flag = save_page(url, "prediction/last_date")
    if(suc_downl_flag == False):
        today_date = today_date - datetime.timedelta(days=1)
    


# In[14]:


def remove_one_letter_word(data):
    words = word_tokenize(str(data))
    new_text = ""
    for w in words:
        if len(w) > 1:
            new_text = new_text + ' ' + w                                                                                                                                                                                                                                                                                             
    return new_text

def convert_lower_case(data):
      return np.char.lower(data)

             
def remove_stop_words(data):
    stop_words = set(stopwords.words('english'))
    stop_stop_words = {"no","not"}
    stop_words = stop_words - stop_stop_words                        

    words = word_tokenize(str(data))

    new_text = ""
    for w in words:
        if w not in stop_words and len(w) > 1:
            new_text = new_text +" "+ w
    return new_text

def remove_punctuation(data):
    symbols = "!\"#$%&()*+—./:;<=>7@[\]^_'{|}~\n"

    for i in range(len(symbols)):
        data = np.char.replace(data, symbols[i], ' ')
        data = np.char.replace(data, "  ", " ")

    data = np.char.replace(data, ',', "")

    return data

def remove_apostrophe(data):
    return np.char.replace(data, "'", "")

def remove_map_line(data):
    return np.char.replace(data, "Click here to expand the map below.", "")

def remove_map_line_v2(data):
    return np.char.replace(data, "Click here to see ISW’s interactive map of the Russian invasion of Ukraine. This map is updated daily alongside the static maps present in this report.", "")

def convert_numbers(data):

    tokens = word_tokenize(str(data))
    new_text = " "
    for w in tokens:
        if w.isdigit():
            if int(w)<1000000000000:
                w = num2words (w)
            else:
                w = ''
        new_text = new_text +" " + w
    new_text = np.char.replace(new_text, "-", " ")
                                                
    return new_text

def stemming(data):
    stemmer= PorterStemmer()

    tokens = word_tokenize(str(data))

    new_text = ""
    for w in tokens:
        new_text = new_text + " " + stemmer.stem(w)
    return new_text

def lemmatizing(data):
    lemmatizer = WordNetLemmatizer()

    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        new_text = new_text + " " + lemmatizer.lemmatize(w)
    return new_text


# In[15]:


def preprocess(data, word_root_algo="lemm"):
    data = remove_map_line(data)
    data = remove_map_line_v2(data)
    data = remove_one_letter_word(data)
    data = convert_lower_case(data)
    data = remove_punctuation(data) #remove comma seperately
    data = remove_apostrophe (data)
    data = remove_stop_words(data)
    data = convert_numbers(data)
    data = stemming(data)
    data = remove_punctuation(data)
    data = convert_numbers (data)

    if word_root_algo == "lemm":
        print ("lennatizing")
        data = lemmatizing(data) #needed again as we need to lemmatize the words
    else:
        print("stemming")
        data = stemming(data) #needed again as we need to stem the words

    data = remove_punctuation(data) #needed again as num2word is giving few hypens and commas fourty-one
    data = remove_stop_words(data) #needed again as num2word is giving stop words 101 - one hundred and one

    return data


# In[16]:





# In[22]:


all_data = []
print(files_by_days)

for file in files_by_days:
    name = file.split(".")[0].lstrip(INPUT_FOLDER)
    name = name.replace("\\", "") 
    print(name)  #
    d = {} 
# Open the HTML file
    with open(file, encoding="utf8") as file:
            soup = BeautifulSoup(file, 'html.parser')

            # Extract the text from the HTML
            text = soup.get_text()

            # Find the index of the first occurrence of "ET"
            index = text.find("ET")

            # Extract the text after the first occurrence of "ET"
            text = text[2+index:]
            index = text.rfind("[1]")
            text =text[:index]
            text = re.sub(r'\[.*?\]', '', text)

            lemm = preprocess(text)
            stemm = preprocess(text, "stemm")
                        
            d = {
                "date":name,
                "text":text,
                "lemm":lemm,
                "stemm":stemm
            }
            all_data.append(d)

df = pd.DataFrame.from_dict(all_data)
df = df.sort_values(by = ['date'])
#df.head(5)


# In[18]:


df.to_csv(f"{OUTPUT_FOLDER}/{OUTPUT_FILE}", sep=";", index=False)


# In[28]:


docs = df['lemm'].tolist()


# In[31]:


len(docs)


# In[38]:





# In[39]:


tfidf = pickle.load(open(f"{MODEL_FOLDER}/{tfidf_transformer_model}_{tfidf_transformer_version}.pkl", "rb"))
cv = pickle.load(open(f"{MODEL_FOLDER}/{count_vectorizer_model}_{count_vectorizer_version}.pkl", "rb"))


# In[49]:


word_count_vector = cv.transform(df["lemm"].values.astype("U"))

# Calculate global tfidf vectors matrix
tfidf_vector = tfidf.transform(word_count_vector)

df_tfidf_vector = pd.DataFrame.sparse.from_spmatrix(tfidf_vector, columns=cv.get_feature_names_out())
#word_count_vector.shape


# In[47]:


#print(df_tfidf_vector)


# In[50]:


with open(f"{OUTPUT_FOLDER}/{OUTPUT_DATA_FILE}.pkl", 'wb') as handle:
    pickle.dump(df_tfidf_vector, handle)


# In[ ]:




