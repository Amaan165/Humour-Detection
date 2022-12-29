


data = {
    "BagofWords" : {
        "Title" : "Bag Of Words",
        "Description": "The bag-of-words model is the most common text classification technique. The frequency (or "
                       "occurrence) of each word is used as a feature to learn the humour classifier. The text is "
                       "converted to numerical data using CountVectorizer() The meaning of the sentence is not taken "
                       "into account.",
        "Accuracy": "88.29 %",

        "ClassificationReport": {
            "Non-Humorous": {
                "precision": "0.90",
                "recall": "0.87",
                "f1-score": "0.88",
                "support": "4967",
            },
            "Humorous": {
                "precision": "0.87",
                "recall": "0.90",
                "f1-score": "0.89",
                "support": "5033",
            },
            "Accuracy": {
                "precision": "",
                "recall": "",
                "f1-score": "0.88",
                "support": "10000",
            },
            "Macro-Avg": {
                "precision": "0.88",
                "recall": "0.88",
                "f1-score": "0.88",
                "support": "10000",
            },
            "Weighted-Avg": {
                "precision": "0.88",
                "recall": "0.88",
                "f1-score": "0.88",
                "support": "10000",
            },
        },

        "ConfusionMatrix": [[4297, 670],[501, 4532]]
    },

    "TFIDF": {
        "Title": "TF-IDF",
        "Description": "The TF-IDF model uses the product of term frequency and inverse document frequency (TF*IDF) "
                       "to find out the importance of a word. It can figure out if a word is relevant to a particular "
                       "document or not. This model uses TfidfVectorizer() to convert text into a matrix of TF-IDF "
                       "features.",
        "Accuracy": "88.27 %",

        "Classification Report": {
            "Non-Humorous": {
                "precision": "0.89",
                "recall": "0.87",
                "f1-score": "0.88",
                "support": "4967",
            },
            "Humorous": {
                "precision": "0.87",
                "recall": "0.90",
                "f1-score": "0.88",
                "support": "5033",
            },
            "Accuracy": {
                "precision": "",
                "recall": "",
                "f1-score": "0.88",
                "support": "10000",
            },
            "Macro-Avg": {
                "precision": "0.88",
                "recall": "0.88",
                "f1-score": "0.88",
                "support": "10000",
            },
            "Weighted-Avg": {
                "precision": "0.88",
                "recall": "0.88",
                "f1-score": "0.88",
                "support": "10000",
            },
        },

        "ConfusionMatrix": [[4318, 649], [524, 4509]]
    },

    "W2V": {
        "Title": "Word2Vec",
        "Description": "The word2vec model creates a word embedding using two methods: Skip Gram and "
                       "CBOW (Common Bag Of Words). This embedding is used to map "
                       "text (or words) to vectors having several dimensions that represent different features. "
                       "This word vector representations act as weights of the neural network which are learned by the model.",
        "Accuracy": "",

        "Classification Report": {
            "Non-Humorous": {
                "precision": "0.76",
                "recall": "0.71",
                "f1-score": "0.73",
                "support": "1956",
            },
            "Humorous": {
                "precision": "0.74",
                "recall": "0.79",
                "f1-score": "0.76",
                "support": "2044",
            },
            "Accuracy": {
                "precision": "",
                "recall": "",
                "f1-score": "0.75",
                "support": "4000",
            },
            "Macro-Avg": {
                "precision": "0.75",
                "recall": "0.75",
                "f1-score": "0.75",
                "support": "4000",
            },
            "Weighted-Avg": {
                "precision": "0.75",
                "recall": "0.75",
                "f1-score": "0.75",
                "support": "4000",
            },
        },

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

        "ConfusionMatrix": [[4318, 649], [524, 4509]]
    },


}

