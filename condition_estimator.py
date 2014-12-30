from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier

import numpy as np
import sys

"""
Conditional Estimator
"""
class ConditionEstimator(BaseEstimator, ClassifierMixin):
    def fit(self, X, y, scikit_estimator):
        matrix = np.array(y)
        sample_num, self.y_num = matrix.shape

        # estimator for first y column
        self.base_estimator = scikit_estimator().fit(X, matrix[:, 0])

        # create estimator for each remaining y columns
        self.condition_estimator = {}
        for y_idx in range(1, self.y_num):
            # dictionary to save X and estimator based on previous y's value 
            self.condition_estimator[y_idx] = {}
            
            # for each sample, predict current y column base on previous y column's output
            for x_idx, row in enumerate(X):
                # get the value of previsou y column
                val = matrix[x_idx][y_idx-1]

                # save corresponding X based on previous y value
                if not val in self.condition_estimator[y_idx]:
                    self.condition_estimator[y_idx][val] = {"X": [], "Y": [], "estimator": None}
                cond_dict = self.condition_estimator[y_idx].get(val)
                cond_dict["X"].append(row)
                cond_dict["Y"].append(matrix[x_idx][y_idx])

            # fit each estimator using corresponding data set
            for val in self.condition_estimator[y_idx].keys():
                sample = self.condition_estimator[y_idx][val]["X"]
                target = self.condition_estimator[y_idx][val]["Y"]
                self.condition_estimator[y_idx][val]["estimator"] = scikit_estimator().fit(sample, target)

        return self

    def predict(self, X):
        y = []

        # get predictions of first y column
        prev_pred = list(self.base_estimator.predict(X))
        y.append(prev_pred)

        # for each remaining y column, predict using corresponding estimator
        for y_idx in range(1, self.y_num):
            predictions = []
            for x_idx, row in enumerate(X):
                # get previous y column's prediction value
                pred = prev_pred[x_idx]
                if pred in self.condition_estimator[y_idx]:
                    # get estimator based on previous y prediction value and predict
                    estimator = self.condition_estimator[y_idx][pred]["estimator"]
                    condition_pred = estimator.predict(row)[0]
                    predictions.append(condition_pred)
                else:
                    print "error: value not in condition_estimator"
                    sys.exit(1)
            y.append(predictions)
            prev_pred = predictions

        return np.matrix(y).transpose()

    """
    Compute conditional prediction probability, if threshold is larger than 0, stop 
    when probality is smaller than threshold
    Return a list of predictions and a list of probabilities correspondingly
    """
    def predict_proba(self, X, threshold=0.0):
        y = [] # 2 dimention prediction list, each row is prediction for a sample
        proba = [] # 2 dimention probability list, each row is probability for a sample
        prev_pred = [] # previous y column prediction, updated each loop
        stop = [] # record if to continue for a sample
        
        # compute the prediction probability using the first level y estimator 
        base_classes = self.base_estimator.classes_
        for x_idx, base_prob in enumerate(self.base_estimator.predict_proba(X)):
            max_prob = -1
            pred = None
            # find prediction class and probability
            for idx, base_class in enumerate(base_classes):
                if base_prob[idx] > max_prob:
                    max_prob = base_prob[idx]
                    pred = base_class
            proba.append(max_prob)
            prev_pred.append(pred)
            y.append([pred])
            stop.append(False)
        
        # for each remaining y column, get prediction class and probability
        for y_idx in range(1, self.y_num):
            predictions = []
            for x_idx, row in enumerate(X):
                pred = prev_pred[x_idx]
                pred_prob = proba[x_idx]
                predictions.append(None)

                # continue predicting if previous prediction probability is larger than threshold
                if (pred_prob > threshold) and (not stop[x_idx]):
                    if pred in self.condition_estimator[y_idx]:
                        estimator = self.condition_estimator[y_idx][pred]["estimator"]
                    
                        est_prob = estimator.predict_proba(row)[0]
                        est_classes = estimator.classes_

                        # get prediction class and probability
                        max_prob = -1
                        condition_pred = None
                        for idx, est_class in enumerate(est_classes):
                            if est_prob[idx] > max_prob:
                                max_prob = est_prob[idx]
                                condition_pred = est_class
                        
                        # if current probability is not larger than threshold,
                        # then mark stop as true for this sample, otherwise add
                        # the prediction to the list and modify probability
                        if proba[x_idx]*max_prob <= threshold:
                            stop[x_idx] = True
                        else: 
                            y[x_idx].append(condition_pred)
                            proba[x_idx] *= max_prob
                        # update previous probability
                        predictions[x_idx] = condition_pred
                    else:
                        print "error: value not in condition_estimator"
                        sys.exit(1)
            prev_pred = predictions

        return y, proba

def main():
    est = ConditionEstimator()
    sk_est = RandomForestClassifier
    est.fit([[1, 2, 3], [2, 3, 4], [5, 6, 7]], [["a", "x", "a"], ["b", "y", "a"], ["a", "y", "b"]], sk_est)

    #print est.predict([[1, 2, 3], [2, 4, 4]])
    
    pred, proba = est.predict_proba([[1, 2, 3], [2, 4, 4]], 0.6)
    print pred
    print proba

if __name__ == "__main__":
    main()
