import pandas as pd
import numpy as np
import argparse
import pickle
from sklearn.model_selection import train_test_split
from utils.constants import (
    DEFAULT_DOC2VEC_MODEL_PATH,
    PROCESSED_FILE_PATH,
    CORPUS_PATH,
)

from gensim.models.doc2vec import Doc2Vec
from extract_topics import LDA
from nn import NeuralNetwork
from preprocess import Preprocessor 
from feature_engineer import FeatureEngineer


INPUT_PATH = PROCESSED_FILE_PATH

def train_neural_network(
    training_size: int
):
    d2v = Doc2Vec.load(DEFAULT_DOC2VEC_MODEL_PATH)  

    df = pd.read_csv(INPUT_PATH)[:training_size]

    preprocessor = Preprocessor()

    X = d2v.wv.vectors[:training_size]

    y , mlb = preprocessor.transform_multilabels(df)
    

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )


    nn = NeuralNetwork(input_size= X_train.shape[1], output_size=y_train.shape[1])

    nn.begin_training(X_train, y_train, X_test, y_test)

    
    with open('assets/mlb.pkl', 'wb') as f:
        pickle.dump(mlb, f)
    


    


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




