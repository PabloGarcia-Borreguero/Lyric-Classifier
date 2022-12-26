import pandas as pd
from utils.constants import (
    DEFAULT_TOPIC_MODEL_PATH,
    DEFAULT_FILE_PATH,
    CORPUS_PATH,
    DEFAULT_TOPIC_MODEL_PATH
)
import argparse
import torch
from preprocess import Preprocessor
from feature_engineer import FeatureEngineer
from extract_topics import LDA
from nn import NeuralNetwork
from gensim.models.ldamodel import LdaModel
from sklearn.preprocessing import MultiLabelBinarizer


DEFAULT_FILE_PATH=DEFAULT_FILE_PATH
DEFAULT_TOPIC_MODEL_PATH=DEFAULT_TOPIC_MODEL_PATH



def classify_songs(
    *kwargs,
)-> None:
    """Trains LDA topic modeller on processed
    song lyrics.

    Args:
        training_size (int): Size of data to 
            use during training.
        num_topics (int): Number of topics to 
            extract during training.
    """

    preprocessor = Preprocessor()
    fe = FeatureEngineer()
    lda = LDA()
    
  
    
    # import input data as p
    df = pd.read_csv(DEFAULT_FILE_PATH)
    df = df.sample(200)

    df_processed = preprocessor.run_preprocessing(df)

    input_corpus = list(df_processed["Output"])
    corpus, _ = fe.construct_dictionary(
        input_corpus
    )
    ldamodel = LdaModel.load(DEFAULT_TOPIC_MODEL_PATH)
    X = lda.get_topic_vectors(df_processed, corpus, ldamodel)

    mlb_labels, mlb = preprocessor.transform_multilabels(df_processed)
    
    nn_model = torch.load("assets/model.pt")
    
    nn_class = NeuralNetwork(input_size= X.shape[1], output_size=mlb_labels.shape[1])
    X = nn_class.scale_and_standardize(X)
    X = torch.from_numpy(X)
    output = nn_model(X)


    return output
    

def cli():
    parser = argparse.ArgumentParser(description="Run topic modelling trainer",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    return parser
    
    
if __name__ == '__main__':
    parser = cli()
    args = parser.parse_args()
    classify_songs(
    )