import requests
import json
import datetime
import os

import re
import pickle
import pytz

import datetime as dt
import pandas as pd
import numpy as np

from dateutil import parser
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

from num2words import num2words

from bs4 import BeautifulSoup

from config import *

# Define a list of month names
__month_names = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december']

def __remove_one_letter_word(data):
    words = word_tokenize(str(data))
    new_text = ""
    for w in words:
        if len(w) > 1:
            new_text = new_text + ' ' + w
    return new_text

def __convert_lower_case(data):
      return np.char.lower(data)

def __remove_stop_words(data):
    stop_words = set(stopwords.words('english'))
    stop_stop_words = {"no","not"}
    stop_words = stop_words - stop_stop_words

    words = word_tokenize(str(data))

    new_text = ""
    for w in words:
        if w not in stop_words and len(w) > 1:
            new_text = new_text +" "+ w
    return new_text

def __remove_punctuation(data):
    symbols = "!\"#$%&()*+—./:;<=>7@[\]^_'{|}~\n"

    for i in range(len(symbols)):
        data = np.char.replace(data, symbols[i], ' ')
        data = np.char.replace(data, "  ", " ")

    data = np.char.replace(data, ',', "")

    return data

def __remove_apostrophe(data):
    return np.char.replace(data, "'", "")

def __remove_map_line(data):
    return np.char.replace(data, "Click here to expand the map below.", "")

def __remove_map_line_v2(data):
    return np.char.replace(data, "Click here to see ISW’s interactive map of the Russian invasion of Ukraine. This map is updated daily alongside the static maps present in this report.", "")

def __convert_numbers(data):

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

def __stemming(data):
    stemmer= PorterStemmer()

    tokens = word_tokenize(str(data))

    new_text = ""
    for w in tokens:
        new_text = new_text + " " + stemmer.stem(w)
    return new_text

def __lemmatizing(data):
    lemmatizer = WordNetLemmatizer()

    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        new_text = new_text + " " + lemmatizer.lemmatize(w)
    return new_text

def __isNaN(num):
    return num != num

def __preprocess(data, word_root_algo="lemm"):
    data = __remove_map_line(data)
    data = __remove_map_line_v2(data)
    data = __remove_one_letter_word(data)
    data = __convert_lower_case(data)
    data = __remove_punctuation(data) #remove comma seperately
    data = __remove_apostrophe (data)
    data = __remove_stop_words(data)
    data = __convert_numbers(data)
    data = __stemming(data)
    data = __remove_punctuation(data)
    data = __convert_numbers (data)

    if word_root_algo == "lemm":
        print ("lennatizing")
        data = __lemmatizing(data) #needed again as we need to lemmatize the words
    else:
        print("stemming")
        data = __stemming(data) #needed again as we need to stem the words

    data = __remove_punctuation(data) #needed again as num2word is giving few hypens and commas fourty-one
    data = __remove_stop_words(data) #needed again as num2word is giving stop words 101 - one hundred and one

    return data

def __get_report_lemm(report_file_path):
    lemm = ''
    # Open the HTML file
    with open(report_file_path, encoding="utf8") as file:
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

        lemm = __preprocess(text)

    return lemm


def __save_page(url, file_name):
    # Downloads html page by given url and saves as file_name

    # Send a GET request to the URL and retrieve the response
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Save the response content to a file
        with open(f"{file_name}.html", "wb") as f:
            f.write(response.content)
        return "Page downloaded successfully."
    else:
        return f"Error downloading page. Status code: {response.status_code}"

def get_report_tfidf_vector(report_file_path):
    # Creates TF-IDF vector for given html of ISW report
    lemm = __get_report_lemm(report_file_path)
    # Load TF-IDF
    tfidf = pickle.load(open(f"{MODEL_FOLDER}/{tfidf_transformer_model}_{tfidf_transformer_version}.pkl", "rb"))
    cv = pickle.load(open(f"{MODEL_FOLDER}/{count_vectorizer_model}_{count_vectorizer_version}.pkl", "rb"))
    # Calculate TD-IDF matrix
    word_count_vector = cv.transform([lemm])
    tfidf_vector = tfidf.transform(word_count_vector)

    # transform tfidf matrix to sparse dataframe
    df_tfidf_vector = pd.DataFrame.sparse.from_spmatrix(tfidf_vector, columns=cv.get_feature_names_out())

    return df_tfidf_vector

def save_by_date(date):
    # Download ISW report from given date as datetime.datetime object

    # Parse month and day
    month_name = __month_names[date.month - 1]
    day = int(date.strftime("%d"))
    name = date.strftime("%Y-%m-%d")

    print(name)
    if os.path.exists(f"./Reports/{name}"):
        return "Page allready downloaded."

    if date.strftime("%Y")== "2022":
        if date == datetime.date(2022, 5, 5):
            url = "https://www.understandingwar.org/backgrounder/russian-campaign-assessment-may-5"
        elif date == datetime.date(2022, 7, 11):
            url = "https://www.understandingwar.org/backgrounder/russian-offensive-campaign-update-july-11"
        elif date == datetime.date(2022, 8, 12):
            url = "https://www.understandingwar.org/backgrounder/russian-offensive-campaign-assessment-august-12-0"
        elif date > datetime.date(2022, 2, 28):
            url = ("https://www.understandingwar.org/backgrounder/russian-offensive-campaign-assessment-%s-%d"%(month_name, day))
        elif date == datetime.date(2022, 2, 24):
            url = "https://www.understandingwar.org/backgrounder/russia-ukraine-warning-update-initial-russian-offensive-campaign-assessment"
        elif date == datetime.date(2022, 2, 25):
            url = "https://www.understandingwar.org/backgrounder/russia-ukraine-warning-update-russian-offensive-campaign-assessment-february-25-2022"
        elif date == datetime.date(2022, 2, 28):
            url = "https://www.understandingwar.org/backgrounder/russian-offensive-campaign-assessment-february-28-2022"
        else:
            url = ("https://www.understandingwar.org/backgrounder/russia-ukraine-warning-update-russian-offensive-campaign-assessment-%s-%d"%(month_name, day))
    else:
        if date == datetime.date(2023, 2, 5):
            url = "https://www.understandingwar.org/backgrounder/russian-offensive-campaign-update-february-5-2023"
        else:
            year = date.strftime("%Y")
            url = ("https://www.understandingwar.org/backgrounder/russian-offensive-campaign-assessment-%s-%d-%s"%(month_name, day, year))
    return __save_page(url, f"Reports/{name}")

def save_all():
    # Downloads all ISW reports from Feb 22 2022 till today

    # Set the start date to February 24, 2022
    start_date = datetime.date(2022, 2, 24)

    # Get the current date
    end_date = datetime.date.today()

    # Define a list of month names
    month_names = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december']

    # Iterate over the dates from the start date to the end date
    for date in (start_date + datetime.timedelta(n) for n in range((end_date - start_date).days)):
        save_by_date(date)

