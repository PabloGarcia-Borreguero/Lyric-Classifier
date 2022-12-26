import pandas as pd
import numpy as np
import argparse
import pickle
from sklearn.model_selection import train_test_split
from utils.constants import (
    DEFAULT_TOPIC_MODEL_PATH,
    PROCESSED_FILE_PATH,
    CORPUS_PATH,
)
from gensim.models.ldamodel import LdaModel
from extract_topics import LDA
from nn import NeuralNetwork
from preprocess import Preprocessor
from feature_engineer import FeatureEngineer


INPUT_PATH = PROCESSED_FILE_PATH

def train_neural_network(
    training_size: int
):
    preprocessor = Preprocessor()
    lda = LDA()
    fe = FeatureEngineer()
    ldamodel = LdaModel.load(DEFAULT_TOPIC_MODEL_PATH)    

    df = pd.read_csv(INPUT_PATH)

    df = df.sample(training_size)
    # clean and preprocess each song lyric
    df_processed = preprocessor.run_preprocessing(df)
    # transform corpus and train LDA
    input_corpus = list(df_processed["Output"])
    corpus, dictionary = fe.construct_dictionary(
        input_corpus
    )
    X = lda.get_topic_vectors(df_processed, corpus, ldamodel)
    y , _ = preprocessor.transform_multilabels(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )


    nn = NeuralNetwork(input_size= X_train.shape[1], output_size=y_train.shape[1])

    trained_nn = nn.begin_training(X_train, y_train, X_test, y_test)


    


def cli():
    parser = argparse.ArgumentParser(description="Run topic modelling trainer",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-t", 
        "--training_size", 
        required=True,
        type=int,
        help="size of training data",
    )
    return parser
    
    
if __name__ == '__main__':
    parser = cli()
    args = parser.parse_args()
    train_neural_network(
        training_size=args.training_size, 
    )




