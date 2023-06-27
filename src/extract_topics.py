"Module containing functionality of Latent Dirichlet Allocation method for Topic Modelling"
import gensim
import pandas as pd



import numpy as np

from preprocess import Preprocessor
from feature_engineer import FeatureEngineer
#from utils.constants import DEFAULT_TOPIC_MODEL_PATH


class LDA():

    def __init__(self, 
        num_topics: int = None,
        passes: int = None,
        ):
        """Initializes LDA class"""
        self.processor = Preprocessor()
        self.fe = FeatureEngineer()
        self.topics = num_topics
        self.passes = passes
        
    
    
    def train_lda(self, corpus, dictionary):
        """_summary_

        Args:
            df (_type_): _description_

        Returns:
            _type_: _description_
        """

        ldamodel = gensim.models.ldamodel.LdaModel(
            corpus=corpus, 
            num_topics = 100, 
            id2word=dictionary, 
            passes=self.passes,
            per_word_topics=True
        )
        ldamodel.save(DEFAULT_TOPIC_MODEL_PATH)

    def get_topic_vectors(self, df, corpus, ldamodel):
        """_summary_

        Args:
            df (_type_): _description_
            corpus (_type_): _description_
            ldamodel (_type_): _description_

        Returns:
            _type_: _description_
        """
        train_vecs = []
        for i in range(len(corpus)):
            top_topics = ldamodel.get_document_topics(corpus[i], minimum_probability=0.0)
            topic_vec = [top_topics[i][1] for i in range(len(top_topics))]
            topic_vec.extend([df.iloc[i].Songs])
            topic_vec.extend([len(df.iloc[i].Output)])
            topic_vec.extend([df.iloc[i].Popularity])
            train_vecs.append(topic_vec)

        return np.asarray(train_vecs)

