import nltk
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class DBOWFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, aggregation, word_vectors=None, word_indices=None,
                 stopwords=True):
        """

        Args:
            aggregation (): the operation to be applied on the word embeddings
                of each doc, in order to obtain the doc representation
            word_vectors (): the word vectors matrix
            word_indices (): the index of the word embedding
            stopwords (): False -> omit stop words
        """
        self.stopwords = stopwords
        self.aggregation = aggregation

        self.word_vectors = word_vectors
        self.word_indices = word_indices

        self.vec_dim = word_vectors[0].size
        self.stops = set(nltk.corpus.stopwords.words('english'))

    def aggregate_vectors(self, vectors, operation):

        if operation == "sum":
            return np.sum(np.array(vectors), axis=0)
        if operation == "mean":
            return np.mean(np.array(vectors), axis=0)
        if operation == "min":
            return np.amin(np.array(vectors), axis=0)
        if operation == "max":
            return np.amax(np.array(vectors), axis=0)
        if operation == "minmax":
            max_vec = np.amax(np.array(vectors), axis=0)
            min_vec = np.amin(np.array(vectors), axis=0)
            return np.hstack([min_vec, max_vec])

    def vectorize(self, X):
        return np.array([self.word_vectors[self.word_indices[w]]
                         for w in X
                         if w in self.word_indices and
                         not (self.stopwords is False and w in self.stops)])

    def transform(self, X, y=None):
        vectorized = [self.vectorize(x) for x in X]
        vecs = [self.aggregate_vectors(v, self.aggregation)
                for v in vectorized]
        return vecs

    def fit(self, X, y=None):
        return self
