from typing import List
from gensim import corpora
from gensim.models import TfidfModel
import pickle

class FeatureEngineer():
    """_summary_
    """
    def __init__(self):
        "Initializes Feature Engineering class"

    def construct_dictionary(self, sentences: List[str]):
        dictionary = corpora.Dictionary(sentences)
        corpus = [dictionary.doc2bow(text) for text in sentences]
        pickle.dump(corpus, open('corpus.pkl', 'wb'))
        dictionary.save('dictionary.gensim')

        return corpus, dictionary

    def vectorize(self, sentences: List[str]):
        vectorizer = TfidfModel()
        dictionary = corpora.Dictionary(sentences)
        corpus = [dictionary.doc2bow(text) for text in sentences]
        X = TfidfModel(corpus)
    
        return X, dictionary

    