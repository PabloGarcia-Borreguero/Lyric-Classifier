import pandas as pd
import numpy as np
from feature_engineer import FeatureEngineer
from preprocess import Preprocessor
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import Memory
import hdbscan


class Clusterer():

    def __init__(self):
        """Initializes Training class"""
        self.processor = Preprocessor()
        self.encoder = FeatureEngineer()
        
    
    def group_embeddings(self, df):
        """_summary_

        Args:
            df (_type_): _description_

        Returns:
            _type_: _description_
        """

        clean_data = self.processor.run_preprocessing(df)

        sentences = clean_data["Output"]

        embeddings = self.encoder.encode_sentences(sentences)

        embeddings = self.encoder.compress_dimensions(embeddings, 15)

        clusters = hdbscan.HDBSCAN(min_cluster_size=10,
                        min_samples=60).fit(embeddings)

        self.projection = self.encoder.compress_dimensions(embeddings, 2)

        self.plot_clusters(clusters)
        
        return set(clusters.labels_)

    def score_clusters(self, clusters, prob_threshold = 0.05):
        """
        Returns the label count and cost of a given cluster supplied from running hdbscan
        """
        
        cluster_labels = clusters.labels_
        label_count = len(np.unique(cluster_labels))
        total_num = len(clusters.labels_)
        cost = (np.count_nonzero(clusters.probabilities_ < prob_threshold)/total_num)
        
        return label_count, cost

    def hdbscan_search(cache_dir, umap_embeddings,  min_min_samples, max_min_samples, 
                   min_min_size, max_min_size):
        mem = Memory(cache_dir)
        search_results = {}
        for min_samples in range(min_min_samples, max_min_samples+1, 25):
            for min_size in range(min_min_size, max_min_size+1, 25):
                model = hdbscan.HDBSCAN(min_cluster_size=min_size,
                                        min_samples=min_samples,
                                        memory=mem).fit(umap_embeddings)
                num_of_clusters = np.max(model.labels_)+1
                outliers = np.unique(model.labels_, return_counts=True)[1][0]
                search_results[(min_size, min_samples)] = (num_of_clusters,
                                                        outliers)
        return search_results

    def plot_clusters(self, clusters):
        color_palette = sns.color_palette('Paired', 12)
        cluster_colors = [color_palette[x] if x >= 0
                        else (0.5, 0.5, 0.5)
                        for x in clusters.labels_]
        cluster_member_colors = [sns.desaturate(x, p) for x, p in
                                zip(cluster_colors, clusters.probabilities_)]
        plt.scatter(*self.projection.T, s=50, linewidth=0, c=cluster_member_colors, alpha=0.25)
        


if __name__ == '__main__':
    df = pd.read_csv("assets/reviews.csv")
    df = df.sample(n=3000)
    clusterer = Clusterer()
    clusterer.group_embeddings(df)
    