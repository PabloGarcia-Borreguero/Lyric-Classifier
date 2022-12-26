import pandas as pd
from typing import List
from numpy import ndarray
from utils.utils import clean_data
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from sklearn.preprocessing import MultiLabelBinarizer

class Preprocessor():
    """Performs language processing of entire pandas dataset
    """

    def __init__(self):
        "Initializes preprocessing class"
        

    def run_preprocessing(self, df: pd.DataFrame
    )-> pd.DataFrame:
        """Performs cleaning, lemmatization, tokenization and 
        transformation into bigrams of each document in the corpus.

        Args:
            df (pd.DataFrame): Raw dataset with song lyrics in text
                column.

        Returns:
            pd.DataFrame: Dataset with transformed output column.
        """
        # remove punctuations
        df_clean = self._clean_dataset(df)
        # removes stopwords and transforms word into
        # its base form.
        df_lemma = self._lemamtize_data(df_clean)
        # tokenizes each word in doc
        df_token = self._tokenize_data(df_lemma)
        # creates bigrams from tokens
        df_bigrams = self._create_bigrams(df_token)

        return df_bigrams

    def transform_multilabels(self, df: pd.DataFrame)-> ndarray:
        """Transforms multilabel specifications for each 
        input into an array of one-hot encode labels.

        Args:
            df (pd.DataFrame): Raw pandas dataset 

        Returns:
            ndarray: One hot ncode matrix for each input.
        """
        mlb = MultiLabelBinarizer()
        # transform Genres column into list
        df["Labels"] = df["Genres"].apply(lambda classes: classes.split(";"))
        # transform multi labels
        mlb_labels =  mlb.fit_transform(df.Labels)

        return mlb_labels, mlb
    
    def _clean_dataset(self, df: pd.DataFrame
    )-> pd.DataFrame:
        """Removes punctuation sings"""
        clean_df = clean_data(df, col="Lyric", clean_col="Output")

        return clean_df

    def _tokenize_data(self, df: pd.DataFrame
    )-> pd.DataFrame:
        """"""
        df["Output"] = df["Output"].apply(
            lambda sent: word_tokenize(sent))

        return df


    def _lemamtize_data(self, df: pd.DataFrame)-> pd.DataFrame:
        """Transforms individual words into its root form (eg
        Chairs-> Chair, loving-> love)"""

        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))

        df["Output"] = df["Output"].apply(
            lambda sent: " ".join(
                lemmatizer.lemmatize(
                    word) for word in sent.split(" ")if word not in stop_words
            )
        )

        return df

    def _create_bigrams(self, df: pd.DataFrame)-> pd.DataFrame:
        """Transforms each token into bigrams with help of pretrained"""

        df["Output"] = df["Output"].apply(
            lambda tokens: self._bigrams(tokens)
        )

        return df
    
    def _bigrams(self, tokens: List[str])-> List[str]:
        """Deploys bigram transformer and transforms 
        token collections into bigrams."""
        bigram = Phrases(tokens, min_count = 5)
        bigram_mod = Phraser(bigram)

        return bigram_mod[tokens]

   
