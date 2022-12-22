from typing import List
from gensim import corpora
from gensim.models import TfidfModel
from sentence_transformers import SentenceTransformer
from umap import UMAP
import pickle

class FeatureEngineer():
    """_summary_
    """
    def __init__(self):
        "Initializes Feature Engineering class"

    def construct_dictionary(self, sentences: List[str]):
        """_summary_

        Args:
            sentences (List[str]): _description_

        Returns:
            _type_: _description_
        """
        dictionary = corpora.Dictionary(sentences)
        corpus = [dictionary.doc2bow(text) for text in sentences]
        tfidf = TfidfModel(corpus)
        corpus_tfidf = tfidf[corpus]
        pickle.dump(corpus, open('corpus.pkl', 'wb'))
        dictionary.save('dictionary.gensim')

        return corpus_tfidf, dictionary

    def encode_sentences(self, sentences: List[str]):
        """_summary_

        Args:
            sentences (List[str]): _description_

        Returns:
            : _description_
        """
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(sentences, batch_size=64)
    
        return embeddings

    def compress_dimensions(self, embeddings, dimensions):
        """_summary_

        Args:
            n_neighbours (_type_): _description_
            n_components (_type_): _description_
            embeddings (_type_): _description_

        Returns:
            _type_: _description_
        """
        compressed_embeddings = (UMAP( n_neighbors=15,
                                n_components=dimensions,
                                metric='cosine')
                            .fit_transform(embeddings))

        return compressed_embeddings


    