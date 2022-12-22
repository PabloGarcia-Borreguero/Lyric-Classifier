"Module containing functionality of Latent Dirichlet Allocation method for Topic Modelling"
import gensim
import pandas as pd
from collections.abc import Iterable
import tqdm


from preprocess import Preprocessor
from feature_engineer import FeatureEngineer
from cluster_docs import Clusterer
from gensim.models.hdpmodel import HdpModel
from gensim.models.callbacks import PerplexityMetric, ConvergenceMetric, CoherenceMetric


class LDA():

    def __init__(self, num_topics: int):
        """Initializes LDA class"""
        self.processor = Preprocessor()
        self.fe = FeatureEngineer()
        self.topics = num_topics
        self.epochs = 20
        
    
    
    def train_lda(self, df):
        """_summary_

        Args:
            df (_type_): _description_

        Returns:
            _type_: _description_
        """
        df_processed = self.processor.run_preprocessing(df)
        corpus, dictionary = self.fe.construct_dictionary(list(df_processed["Output"]))

        ldamodel = gensim.models.ldamodel.LdaModel(
            corpus=corpus, 
            num_topics = 100, 
            id2word=dictionary, 
            passes=self.epochs,
            per_word_topics=True
        )
        ldamodel.save('assets/model.gensim')

        embeddings = self.get_topic_vectors(df_processed, corpus, ldamodel)

        return embeddings

    def get_topic_vectors(self, df, corpus, ldamodel):
        train_vecs = []
        for i in range(len(corpus)):
            top_topics = ldamodel.get_document_topics(corpus[i], minimum_probability=0.0)
            topic_vec = [top_topics[i][1] for i in range(len(top_topics))]
            topic_vec.extend([df.iloc[i].Songs])
            topic_vec.extend([len(df.iloc[i].Output)])
            topic_vec.extend([df.iloc[i].Popularity])
            train_vecs.append(topic_vec)

        return train_vecs

if __name__ == '__main__':
    df = pd.read_csv("assets/lyrics.csv")
    df = df.sample(100)
    #clusterer = Clusterer()
    #topics = clusterer.group_embeddings(df)
    trainer = LDA(num_topics=150)
    trainer.train_lda(df)