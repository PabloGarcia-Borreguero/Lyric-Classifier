import gensim
import pandas as pd
from preprocess import Preprocessor
from feature_engineer import FeatureEngineer


class Train():

    def __init__(self, num_topics: int):
        """Initializes Training class"""
        self.processor = Preprocessor()
        self.fe = FeatureEngineer()
        self.topics = num_topics
        self.epochs = 15
    
    
    def train_lda(self, df):
        df_processed = self.processor.run_preprocessing(df)
        corpus, dictionary = self.fe.construct_dictionary(list(df_processed["Output"]))
        ldamodel = gensim.models.ldamodel.LdaModel(
            corpus, 
            num_topics = self.topics, 
            id2word=dictionary, 
            passes=self.epochs
        )
        ldamodel.save('assets/model.gensim')

    
        return ldamodel

if __name__ == '__main__':
    df = pd.read_csv("assets/reviews.csv")
    df = df.iloc[:100]
    trainer = Train(num_topics=6)
    trainer.train_lda(df)