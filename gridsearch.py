#!/usr/bin/env python

from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.grid_search import GridSearchCV

import operator


class QuickGridsearch(object):
    def __init__(self, pipeline, params, params_sizes, data_sizes, x_train, y_train):
        self.pipeline = pipeline
        self.params = params
        self.params_sizes = params_sizes
        self.data_sizes = data_sizes
        self.x_train = x_train
        self.y_train = y_train
        self.model = None

    def run(self):
        total = self.x_train.shape[0]
        for idx, size in enumerate(self.data_sizes):
            if size <= 1: # if size<=1, then it is percentage
                train_num = int(total * size)
            else:
                train_num = int(size)

            self.model = GridSearchCV(self.pipeline, self.params, scoring='accuracy', n_jobs=5, cv=5, verbose=1)
            # use partial training data to fit the model
            self.model.fit(self.x_train[0:train_num, :], self.y_train[0:train_num])
    
            # rank paramater combination by score
            
            key_to_score = {}
            key_to_param = {}

            for key, grid_score in enumerate(self.model.grid_scores_):
                key_to_score[key] = grid_score.mean_validation_score
                key_to_param[key] = grid_score.parameters

            sorted_key = sorted(key_to_score.items(), key=operator.itemgetter(1), reverse=True)

            # construct new params based on top ranking parameter combinations
            if idx != (len(self.data_sizes)-1):
                self.params = {}
                for i in range(self.params_sizes[idx]):
                    key = sorted_key[i][0] # get key
                    for param_name in key_to_param[key]:
                        if param_name in self.params:
                            self.params[param_name].append(key_to_param[key][param_name])
                        else:
                            self.params[param_name] = [key_to_param[key][param_name]]
                
                for param_name in self.params:
                    self.params[param_name] = list(set(self.params[param_name]))

def main():
    digits = load_digits()
    X_ = digits['images']
    m  = X_.shape[0]
    X  = X_.reshape(m, -1)
    y  = digits['target']
    x_train, x_test, y_train, y_test = train_test_split(X, y)

    pipeline = Pipeline(steps=[
        ('GBC', GradientBoostingClassifier(random_state=1))
    ])

    data_sizes = [400, 500, 600]

    params = {
        'GBC__n_estimators'            : [5, 10, 15],
        'GBC__max_depth'               : [10, 20, 30],
        'GBC__min_samples_split'       : [10, 15, 20],
    }

    params_sizes = [10, 5] # length of params_sizes is one less than data_sizes

    gs = QuickGridsearch(pipeline, params, params_sizes, data_sizes, x_train, y_train)
    gs.run()

    print gs.model.best_estimator_
    print gs.model.best_params_

if __name__ == "__main__":
    main()

