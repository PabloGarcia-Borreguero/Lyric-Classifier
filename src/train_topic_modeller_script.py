import pandas as pd
from utils.constants import (
    DEFAULT_FILE_PATH,
    PROCESSED_FILE_PATH
)
from feature_engineer import FeatureEngineer
from preprocess import Preprocessor
from extract_topics import LDA
import argparse

INPUT_PATH = DEFAULT_FILE_PATH

PROCESSED_FILE_PATH = PROCESSED_FILE_PATH

def train_topic_modeller(
    training_size: int = 3000, 
    num_topics: int = 100,
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
    lda = LDA(num_topics=num_topics, passes = 15)
    # import input data as pandas dataframe
    df = pd.read_csv(INPUT_PATH)
    df = df.iloc[:training_size]
    # clean and preprocess each song lyric
    df_processed = preprocessor.run_preprocessing(df)
    df_processed.to_csv(PROCESSED_FILE_PATH)
    # transform corpus and train LDA
    input_corpus = list(df_processed["Output"])
    corpus, dictionary = fe.construct_dictionary(
        input_corpus
    )
    lda.train_lda(corpus, dictionary)


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
    parser.add_argument(
        "-n", 
        "--num_topics",
        required=True, 
        type=int,
        help="topic extraction number",
    )
    return parser
    
    
if __name__ == '__main__':
    parser = cli()
    args = parser.parse_args()
    train_topic_modeller(
        training_size=args.training_size, 
        num_topics= args.num_topics
    )


