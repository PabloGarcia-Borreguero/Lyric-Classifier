import pandas as pd
from utils.constants import (
    DEFAULT_FILE_PATH,
    CORPUS_PATH,
    DEFAULT_DOC2VEC_MODEL_PATH
)
import argparse
import torch
from preprocess import Preprocessor
from feature_engineer import FeatureEngineer
from nn import NeuralNetwork
from gensim.models.doc2vec import Doc2Vec
from sklearn.preprocessing import MultiLabelBinarizer
import pickle


DEFAULT_FILE_PATH=DEFAULT_FILE_PATH


class Classifier():

    def __init__(self):
        "Initializes classifier class"
        self.preprocessor = Preprocessor()
        self.fe = FeatureEngineer()
        self.doc2vec = Doc2Vec.load(DEFAULT_DOC2VEC_MODEL_PATH)
        self.nn_model = torch.load("assets/model.pt")
        with open('assets\mlb.pkl', 'rb') as f:
            self.mlb = pickle.load(f)
    
    def classify_song(self, lyrics: str, songs: int, popularity: int):
        """_summary_

        Args:
            lyrics (str): _description_
            songs (int): _description_
            popularity (int): _description_

        Returns:
            _type_: _description_
        """
        lyrics_df = pd.DataFrame(
            data=[[songs, lyrics, popularity]], 
            columns=["Songs", "Lyric", "Popularity" ]
        )
        df_processed = self.preprocessor.run_preprocessing(lyrics_df)

        X = self.doc2vec.infer_vector(df_processed.Output.values[0])

        nn_class = NeuralNetwork(input_size= X.shape[0], output_size=self.nn_model.output_dim)

        X = nn_class.scale_and_standardize(X)
        X = torch.from_numpy(X)
        output = self.nn_model(X)
        
        pred =  (output>0.1).float().detach().numpy()

        classes  = self.mlb.inverse_transform(pred)
        return classes 

    def classify_data(self, 
        data: pd.DataFrame
    )-> None:
        """Trains LDA topic modeller on processed
        song lyrics.

        Args:
            training_size (int): Size of data to 
                use during training.
            num_topics (int): Number of topics to 
                extract during training.
        """
    
        df = pd.read_csv(DEFAULT_FILE_PATH)
        df = df.sample(200)

        df_processed = self.preprocessor.run_preprocessing(df)

        input_corpus = list(df_processed["Output"])
        corpus, _ = self.fe.construct_dictionary(
            input_corpus
        )
        ldamodel = LdaModel.load(DEFAULT_TOPIC_MODEL_PATH)
        X = self.lda.get_topic_vectors(df_processed, corpus, ldamodel)

        mlb_labels, mlb = self.preprocessor.transform_multilabels(df_processed)
        
        nn_model = torch.load("assets/model.pt")
        
        nn_class = NeuralNetwork(input_size= X.shape[1], output_size=mlb_labels.shape[1])
        X = nn_class.scale_and_standardize(X)
        X = torch.from_numpy(X)
        output = nn_model(X)

        genres = self.mlb.inverse_transform(output)


        return genres
    

def cli():
    parser = argparse.ArgumentParser(description="Run topic modelling trainer",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    return parser
    
    
if __name__ == '__main__':
    parser = cli()
    args = parser.parse_args()
    classifier = Classifier()
    classifier.classify_song(lyrics=
    """
    On the road, in the air
Across the sea, across the land
Going Everywhere, meeting everyone
Around the globe, around your head
Across Your soul, all through your brain
Meeting everyone, going everywhere...

Blessed are the travelers, those who belong nowhere...
On the road again

Traveling further than my thoughts can go...
On the road again
Searching, Finding
Forgetting, remembering
How it goes...

Different cultures,
Different traditions
Different languages, different approaches Of the World...
On the road

World within our eyes, confronted by the outside...
On the road

Far - of destination, constant vacation...
On the road again

Searching, finding
Forgetting, remembering
How it goes...
Introspective Trip
Initiative voyages
All along...

On the road, in the air
Across the Sea, across the land
Going everywhere, meeting everyone
Around the globe, Around your head
Across your soul, all through your brain
Meeting Everyone, going everywhere...""", 
songs= 90.0, 
popularity=0.0
)