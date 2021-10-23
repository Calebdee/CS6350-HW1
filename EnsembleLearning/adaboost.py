import sys
import pandas as pd
import math
import numpy as np
from sklearn.preprocessing import LabelBinarizer, LabelEncoder

from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from sklearn.tree import DecisionTreeClassifier

from sklearn import preprocessing 

def main():
    args = sys.argv[1:]
    if len(args) != 2:
        print("well thats just not ok")

    train = pd.read_csv(args[0], header=None)
    train =  train.apply(preprocessing.LabelEncoder().fit_transform)
    test = pd.read_csv(args[1], header=None)
    #test = np.array(test)
    train_test = np.array(train)
    train_x = pd.DataFrame(train.iloc[:, :-1])
    train_y = pd.DataFrame(train.iloc[:, -1])
    test_x = pd.DataFrame(test.iloc[:, :-1])
    test_y = pd.DataFrame(test.iloc[:, -1])
    test_x = test_x.apply(LabelEncoder().fit_transform)
    train_x = train_x.apply(LabelEncoder().fit_transform)
    types = ["entropy", "gini", "majorityError"]

    ada = AdaBoost()
    myAda = ada.fit(train_x, train_y)
    ada.score(test_x, test_y)

class AdaBoost:
    
    def __init__(self):
        self.alphas = []
        self.Model = []
        self.M = None
        self.training_errors = []

    def fit(self, X, y, M = 100):
        self.alphas = [] 
        self.training_errors = []
        self.M = M
        y = y.to_numpy()

        for m in range(0, M):
            if m == 0:
                w_i = np.ones(len(y)) * 1 / len(y)

            else:
                w_i = self.update_weights(w_i, alpha_m, y, y_pred)

            model = DecisionTreeClassifier(max_depth = 1)
            model.fit(X, y, sample_weight = w_i)
            
            y_pred = model.predict(X)
            
            self.Model.append(model)

            error_m = self.calculate_error(y, y_pred, w_i)

            self.training_errors.append(error_m)

            alpha_m = self.calculate_alpha(error_m)
            self.alphas.append(alpha_m)
            
    def score(self, X, y):
        model = self.Model[self.M-1]
        return model.score(X, y)

    def calculate_error(self, y, y_pred, w_i):
        sum = 0
        for i in range(len(y)):
        	if y[i] != y_pred[i]:
        		sum += w_i[i]
        return sum

    def calculate_alpha(self, error):
        return 0.5 * np.log((1 - error) / error)

    def update_weights(self, w_i, alpha, y, y_pred):
        wt_i = []
        for i in range(len(y)):
        	w_i[i] += w_i[i]  * math.exp(-1*alpha * (y[i] * y_pred[i]))
        w_i = (w_i / sum(w_i))
        return w_i
