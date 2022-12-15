import pandas as pd
from utils.utils import clean_data
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

class Preprocessor():
    """Preprocessing of entire dataset
    """

    def __init__(self):
        "Initializes preprocessing class"
        return None

    def run_preprocessing(self, df: pd.DataFrame):
    
        df_clean = self.clean_dataset(df)
        df_lemma = self.lemamtize_data(df_clean)
        df_token = self.tokenize_data(df_lemma)

        return df_token
    
    def clean_dataset(self, df: pd.DataFrame):
        """Cleans raw dataset string

        Args:
            df (pd.DataFrame): Dataset

        Returns:
            _type_: _description_
        """
        clean_df = clean_data(df, col="Text", clean_col="Output")

        return clean_df
    
    def stem_data(self, df: pd.DataFrame):
         
        """_summary_

        Args:
            df (_type_): _description_
            clean_col (_type_): _description_

        Returns:
            _type_: _description_
        """
        stop_words = set(stopwords.words('english'))
        stemmer = SnowballStemmer('english')
        # remove stopwords and get the stem
        df["Output"] = df["Output"].apply(lambda x: ' '.join(stemmer.stem(text) for text in x.split() if text not in stop_words))

        return df

    def tokenize_data(self, df: pd.DataFrame):
        """_summary_

        Args:
            df (pd.DataFrame): _description_

        Returns:
            _type_: _description_
        """

        df["Output"] = df["Output"].apply(lambda sent: word_tokenize(sent))

        return df


    def lemamtize_data(self, df: pd.DataFrame):
        """_summary_

        Args:
            df (pd.DataFrame): _description_

        Returns:
            _type_: _description_
        """

        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))

        df["Output"] = df["Output"].apply(lambda sent: " ".join(lemmatizer.lemmatize(word) for word in sent.split(" ")if word not in stop_words))

        return df