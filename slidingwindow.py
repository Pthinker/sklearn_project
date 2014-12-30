from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.utils import atleast2d_or_csr

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB


class SlidingWindowEstimator(BaseEstimator, ClassifierMixin):
    #PADDING = "_PADDING_"
    PADDING = -1
    
    def __init__(self, size=1, padding=False):
        self.size = size
        self.padding = padding

    def fit(self, X, Y, scikit_estimator):
        self.estimator = scikit_estimator().fit(X, Y)

        return self
    
    def predict(self, X):
        mapping = {} # save mapping from original X to transformed X

        for idx, row in enumerate(X):
            if len(row) < self.size:
                # if padding is true, then using defined padding string to fill short sample
                if self.padding:
                    row.extend([self.PADDING] * (self.size - len(row)))
                    mapping[idx] = [row]
                else:
                    mapping[idx] = None
            else:
                arr = [row[i:i+(self.size)] for i in xrange(len(row)-self.size+1)]
                mapping[idx] = []
                for new_row in arr:
                    mapping[idx].append(new_row)

        predictions = []
        probabilities = []
        class_labels = self.estimator.classes_
        for i in xrange(len(X)):
            if mapping[i] is None:
                predictions.append(float("NaN"))
            else:
                max = -1.0
                orig_pred = None

                for new_x in mapping[i]:
                    max_prob = -1.0
                    pred = None
                    prob = self.estimator.predict_proba(new_x)[0]
                    for idx, label in enumerate(class_labels):
                        if prob[idx] > max_prob:
                            max_prob = prob[idx]
                            pred = label
                    if max_prob > max:
                        max = max_prob
                        orig_pred = pred

                predictions.append(orig_pred)
                probabilities.append(max)

        return predictions, probabilities
    
    '''
    Using sliding window method to transform training data X and Y
    Return transformed X and Y
    '''
    def transform(self, X, Y):
        atleast2d_or_csr(X)

        self.X_transformed = []
        self.Y_transformed = []

        for idx, row in enumerate(X):
            if len(row) < self.size:
                # if padding is true, then using defined padding string to fill short sample
                if self.padding:
                    row.extend([self.PADDING] * (self.size - len(row)))
                    self.X_transformed.append(row)
                    self.Y_transformed.append(Y[idx])
            else:
                arr = [row[i:i+(self.size)] for i in xrange(len(row)-self.size+1)]
                for new_row in arr:
                    self.X_transformed.append(new_row)
                    self.Y_transformed.append(Y[idx])
        return self.X_transformed, self.Y_transformed


def main():
    #X = [["US","banana","November", "test"], ["Bob","ball"], ["soccer","basketball","New York"], ['a']]
    X = [[1,2,3,4], [1,2,3], [1,4], [2,3], [1]]
    Y = [1, 2, 3, 4]
    
    sw = SlidingWindowEstimator(2, padding=False)
    X_new, Y_new = sw.transform(X, Y)

    sk_est = RandomForestClassifier
    sw.fit(X_new, Y_new, sk_est)

    print sw.predict([[1,2,3,4], [1,2,3]])


if __name__ == "__main__":
    main()

