from typing import List
from gensim import corpora
from gensim.models import TfidfModel
from utils.constants import (
    CORPUS_PATH,
    DICT_PATH
)
import pickle



class FeatureEngineer():
    """_summary_
    """

    CORPUS_PATH = CORPUS_PATH
    DICT_PATH = DICT_PATH
    
    def __init__(self):
        "Initializes Feature Engineering class"

    def construct_dictionary(self, sentences: List[str]):
        """_summary_

        Args:
            sentences (List[str]): _description_

        Returns:
            _type_: _description_
        """
        dictionary = corpora.Dictionary(sentences)
        corpus = [dictionary.doc2bow(text) for text in sentences]
        tfidf = TfidfModel(corpus)
        corpus_tfidf = tfidf[corpus]
        pickle.dump(corpus, open(self.CORPUS_PATH, 'wb'))
        dictionary.save(self.DICT_PATH)

        return corpus_tfidf, dictionary



    