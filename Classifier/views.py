from django.shortcuts import render
import gensim.models
import pickle
import numpy as np
import pandas as pd
import re

import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import gensim
from sklearn.preprocessing import MinMaxScaler

data = {
    "BagofWords": {
        "Title": "Bag Of Words",
        "Description": "The bag-of-words model is the most common text classification technique. The frequency (or "
                       "occurrence) of each word is used as a feature to learn the humour classifier. The text is "
                       "converted to numerical data using a Count Vectorizer. The meaning of the sentence is not taken "
                       "into account.",
        "Accuracy": "88.29 %",

        "ClassificationReport": [
            ['0.90', '0.87', '0.88', '4967'],
            ['0.87', '0.90', '0.89', '5033'],
            [0, 0, '0.88', '10000'],
            ['0.88', ' 0.88', '0.88', '10000'],
            ['0.88', '0.88', '0.88', '10000']
        ],

        "ConfusionMatrix": [[4297, 670], [501, 4532]]
    },

    "TFIDF": {
        "Title": "TF-IDF",
        "Description": "The TF-IDF model uses the product of term frequency and inverse document frequency (TF*IDF) "
                       "to find out the importance of a word. It can figure out if a word is relevant to a particular "
                       "document or not. This model uses TfidfVectorizer() to convert text into a matrix of TF-IDF "
                       "features.",
        "Accuracy": "88.27 %",

        "ClassificationReport": [
            ['0.89', '0.87', '0.88', '4967'],
            ['0.87', '0.90', '0.88', '5033'],
            [0, 0, '0.88', '10000'],
            ['0.88', ' 0.88', '0.88', '10000'],
            ['0.88', '0.88', '0.88', '10000']
        ],

        "ConfusionMatrix": [[4318, 649], [524, 4509]]
    },

    "W2V": {
        "Title": "Word2Vec",
        "Description": "The Word2Vec model creates a word embedding using two methods: Skip Gram and "
                       "CBOW (Common Bag Of Words). This embedding is used to map "
                       "text (or words) to vectors having several dimensions that represent different features. "
                       "This word vector representations act as weights of the neural network which are "
                       "learned by the model.",
        "Accuracy": "76.25 %",

        "ClassificationReport": [
            ['0.76', '0.71', '0.73', '1956'],
            ['0.74', '0.79', '0.76', '2044'],
            [0, 0, '0.75', '4000'],
            ['0.75', '0.75', '0.75', '4000'],
            ['0.75', '0.75', '0.75', '4000']]
        ,

        "ConfusionMatrix": [[1389, 567], [438, 1606]]
    },

    "GloVe": {
        "Title": "",
        "Description": "",
        "Accuracy": "",

        "Classification Report": {
            "Non-Humorous": {
                "precision": "",
                "recall": "",
                "f1-score": "",
                "support": "",
            },
            "Humorous": {
                "precision": "",
                "recall": "",
                "f1-score": "",
                "support": "",
            },
            "Accuracy": {
                "precision": "",
                "recall": "",
                "f1-score": "",
                "support": "",
            },
            "Micro-Avg": {
                "precision": "",
                "recall": "",
                "f1-score": "",
                "support": "",
            },
            "Weighted-Avg": {
                "precision": "",
                "recall": "",
                "f1-score": "",
                "support": "",
            },
        },

        "ConfusionMatrix": [[0, 0], [0, 0]]
    },

}



google_model = gensim.models.KeyedVectors.load_word2vec_format("static/models/GoogleNews-vectors-negative300.bin", binary=True)

def clean_text(txt):
    # cleans text, removes stopwords(are , you , me ,is) , accepts a list of strings, returns list of a string

    cleaned_text = list()
    lines = txt
    for text in lines:
        text = text.lower()
        text = re.sub(r"[,.\"!@#$%^&*(){}?/;`~:<>+=-]", "", text)
        tokens = word_tokenize(text)
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in tokens]
        words = [word for word in stripped if word.isalpha()]
        stop_words = set(stopwords.words("english"))
        stop_words.discard("not")
        PS = PorterStemmer()

        words = [PS.stem(w) for w in words if not w in stop_words]
        words = ' '.join(words)
        cleaned_text.append(words)
    return cleaned_text


def BagWords(input):
    data_processed = pd.read_csv("static/models/processed_data(50k).csv", delimiter=',')
    data_processed.columns = ["text", "humour"]

    CV = pickle.load(open('static/models/BagOfWords50K_CV.pkl', 'rb'))
    model = pickle.load(open('static/models/BagOfWords_model.pkl', 'rb'))

    XTest = CV.transform(input)
    pred = model.predict(XTest)

    if pred[0] == 0:
        return "Not Humorous"
    else:
        return "Humorous"


def TFIDF(input):
    data_processed = pd.read_csv("static/models/processed_data(50k).csv", delimiter=',')
    data_processed.columns = ["text", "humour"]

    TV = pickle.load(open('static/models/TF-IDF50K_TV.pkl', 'rb'))
    model = pickle.load(open('static/models/TF-IDF_model.pkl', 'rb'))

    XTest = TV.transform(input)
    pred = model.predict(XTest)

    if pred[0] == 0:
        return "Not Humorous"
    else:
        return "Humorous"


def W2C(input):
    data_processed = pd.read_csv("static/models/processed_data(20k).csv", delimiter=',')
    data_processed.columns = ["text", "humour"]

    vectoriser = pickle.load(open('static/models/Word2Vec_CV.pkl', 'rb'))
    model = pickle.load(open('static/models/Word2Vec_MNB_model.pkl', 'rb'))
    Data = pickle.load(open('static/models/Word2Vec_Data.pkl', 'rb'))

    X = vectoriser.transform(data_processed['text'].values)
    CountVectorizedData = pd.DataFrame(X.toarray(), columns=vectoriser.get_feature_names_out())
    CountVectorizedData['Class'] = data_processed['humour']
    WordsVocab = CountVectorizedData.columns[:-1]

    X = Data[Data.columns[:-1]].values  # X

    # Feature Scaling, to create scaler to be used for user input
    X_Scaler = MinMaxScaler(feature_range=(0, 1))
    X_Scaler_fit = X_Scaler.fit(X)
    X = X_Scaler_fit.transform(X)

    input = FunctionText2Vec(input, vectoriser, WordsVocab)
    input = X_Scaler_fit.transform(input)
    pred = model.predict(input)

    if pred[0] == 0:
        return "Not Humorous"
    else:
        return "Humorous"


def FunctionText2Vec(inpTextData, vectoriser, WordsVocab):
    # Converting the text to numeric data
    X = vectoriser.transform(inpTextData)
    CountVecData = pd.DataFrame(X.toarray(), columns=vectoriser.get_feature_names_out())

    # Creating empty dataframe to hold sentences
    W2Vec_Data = pd.DataFrame()

    # Looping through each row for the data
    for i in range(CountVecData.shape[0]):

        # initiating a sentence with all zeros
        Sentence = np.zeros(300)

        for word in WordsVocab[CountVecData.iloc[i, :] >= 1]:
            if word in google_model.key_to_index.keys():
                Sentence = Sentence + google_model[word]

        W2Vec_Data = W2Vec_Data.append(pd.DataFrame([Sentence]))
    return (W2Vec_Data)



def make_img(output):
    if output == "Not Humorous":
        im = "nonhumour.jpg"
    else:
        im = "humour.jpg"
    return im



def index(request):
    if request.method == 'GET':
        return render(request, 'index.html', {
            'output': '',
            'output_image': "",
            'user_data': '',
            'mode': 'BagofWords',
            'cleaned_input': ''
        })

    mode = request.POST['mode']
    passer = request.POST['pred-input']
    input = clean_text([request.POST['pred-input']])

    if mode == 'BagofWords':
        output = BagWords(input)
        im = make_img(output)
        return render(request, 'index.html', {
            'output': output,
            'output_image': im,
            'mode': mode,
            'user_data': passer,
            'cleaned_input': input[0]
        })

    if mode == "TFIDF":
        output = TFIDF(input)
        im = make_img(output)
        return render(request, 'index.html', {
            'output': output,
            'output_image': im,
            'mode': mode,
            'user_data': passer,
            'cleaned_input': input[0]
        })

    if mode == "W2V":
        output = W2C(input)
        im = make_img(output)
        return render(request, 'index.html', {
            'output': output,
            'output_image': im,
            'mode': mode,
            'user_data': passer,
            'cleaned_input': input[0]
        })

    return render(request, 'index.html', {
        'output_image': "",
        'mode': mode,
        'user_data': passer,
        'output': 'NONE'
    })


def about(request):
    return render(request, 'about.html')


def analysis(request):
    if request.method == 'GET':
        return render(request, 'analysis.html', {'title': 'none'})

    mode = request.POST['analysisdata']

    if (mode == 'BagofWords') or (mode == "TFIDF") or (mode == "W2V") :
        return render(request, 'analysis.html', {
            'Title': data[mode]["Title"],
            'Description': data[mode]["Description"],
            'Accuracy': data[mode]["Accuracy"],
            'NHP': data[mode]["ClassificationReport"][0][0],
            'NHR': data[mode]["ClassificationReport"][0][1],
            'NHF': data[mode]["ClassificationReport"][0][2],
            'NHS': data[mode]["ClassificationReport"][0][3],

            'HP': data[mode]["ClassificationReport"][1][0],
            'HR': data[mode]["ClassificationReport"][1][1],
            'HF': data[mode]["ClassificationReport"][1][2],
            'HS': data[mode]["ClassificationReport"][1][3],

            'AF': data[mode]["ClassificationReport"][2][2],
            'AS': data[mode]["ClassificationReport"][2][3],

            'MP': data[mode]["ClassificationReport"][3][0],
            'MR': data[mode]["ClassificationReport"][3][1],
            'MF': data[mode]["ClassificationReport"][3][2],
            'MS': data[mode]["ClassificationReport"][3][3],

            'WP': data[mode]["ClassificationReport"][4][0],
            'WR': data[mode]["ClassificationReport"][4][1],
            'WF': data[mode]["ClassificationReport"][4][2],
            'WS': data[mode]["ClassificationReport"][4][3],

            'TP': data[mode]["ConfusionMatrix"][0][0],
            'FP': data[mode]["ConfusionMatrix"][0][1],
            'FN': data[mode]["ConfusionMatrix"][1][0],
            'TN': data[mode]["ConfusionMatrix"][1][1],

            'analysis_data': mode,
        })
