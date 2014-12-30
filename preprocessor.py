from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.utils import atleast2d_or_csr

import ner


class Preprocessor(BaseEstimator, TransformerMixin):
    """
    Preprocessor that filters strings based on wether or not they are words of 
    a certain *part of speech*
    """
    def __init__(self, entity_type=['LOCATION', 'PERSON']):
        self.entity_type = entity_type

    def fit(self, X, y):
        """Do nothing and return the estimator unchanged
        This method is just there to implement the usual API and hence
        work in pipelines.
        """
        atleast2d_or_csr(X)
        return self

    def transform(self, X, y=None, copy=None):
        """
        Parameters
        ----------
        X : array or scipy.sparse matrix with shape [n_samples, n_features]
            The data to normalize, row by row. scipy.sparse matrices should be
            in CSR format to avoid an un-necessary copy.
        """
        atleast2d_or_csr(X)

        tagger = ner.SocketNER(host='localhost', port=8080)
        
        self.X_transformed = []

        for row in X:
            new_row = []
            for entry in row:
                remove = False
                result = tagger.get_entities(entry)
                for key in result:
                    if key in self.entity_type:
                        remove = True
                        break
                if not remove:
                    new_row.append(entry)
            self.X_transformed.append(new_row)

        return self.X_transformed


def main():
    X = [["US","banana","November"], ["Bob","x","ball"], ["soccer","basketball","New York"]]
    P = Preprocessor()
    print P.transform(X)

if __name__ == "__main__":
    main()



