#!/usr/bin/env python

from sklearn.pipeline import Pipeline
from slidingwindow_preprocessor import SlidingWindowPreprocessor

def main():
    X = [["US","banana","November", "test"], ["Bob","ball"], ["soccer","basketball","New York"], ['a']]
    Y = [1, 2, 3, 4]
    P = SlidingWindowPreprocessor(2, padding=False)

    print P.transform(X, Y)

if __name__ == "__main__":
    main()

