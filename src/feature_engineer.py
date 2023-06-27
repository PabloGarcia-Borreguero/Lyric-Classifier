from typing import List
from gensim import corpora
from gensim.models import TfidfModel
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
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

    def execute_doc2vec(self, docs: List[str]):
        """_summary_

        Args:
            sentences (List[str]): _description_
        """
        documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(docs)]
        d2v = Doc2Vec(
            documents=documents,
            window=10,
            vector_size=50,
            min_count=1,
            workers=-1,
            epochs=10
        )
        
        return d2v


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



    