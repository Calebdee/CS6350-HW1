import pylab as pl
import sys
import pandas as pd
import math
from sklearn.utils import shuffle
import numpy as np
from numpy import linalg

def main():
    args = sys.argv[1:]
    train = pd.read_csv(args[0], header=None)
    test = pd.read_csv(args[1], header=None)

    train_x = train.iloc[: , :-1]
    train_y = train.iloc[: , -1]
    train_y = np.where(train_y > 0, 1, -1)

    test_x = test.iloc[: , :-1]
    temp_y = test.iloc[: , -1]
    test_y = np.where(temp_y > 0, 1, -1)
    

    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)
    
    gamma = [0.1, 0.5, 1, 5, 100]
    for gam in gamma:
        kp = KernelPerceptron(kernel="gaussian", gamma=gam)
        kp.fit(train_x, train_y)
        preds = kp.predict(train_x)
        preds2 = kp.predict(test_x)
        correct = np.sum(preds == train_y)
        correct2 = np.sum(preds2 == test_y)
        print("Gamma of " + str(gam))
        print("Training Accuracy of: " + str(correct / len(preds)))
        print("Testign Accuracy of: " + str(correct2 / len(preds2)))
        print("======================================")


class KernelPerceptron():
    def __init__(self, kernel, gamma, T=1):
        self.kernel = kernel
        self.T = T
        self.gamma = gamma
        self.alpha = 5

    def fit(self, X, y):
        threshold = 1e-10
        self.alpha = np.zeros(X.shape[0], dtype=np.float64)

        K = np.zeros((X.shape[0], X.shape[0]))
        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                if self.kernel == "gaussian":
                    K[i,j] = self.gaussian(X[i], X[j], self.gamma)
                elif self.kernel == "linear":
                    K[i,j] = self.linear(X[i], X[j])

        for t in range(self.T):
            for i in range(X.shape[0]):
                if np.sign(np.sum(K[:,i] * self.alpha * y)) != y[i]:
                    self.alpha[i] += 1.0

        support_vectors = self.alpha > threshold
        ind = np.arange(len(self.alpha))[support_vectors]
        self.alpha = self.alpha[support_vectors]
        self.support_vectors = X[support_vectors]
        self.support_vectors_y = y[support_vectors]

    def dual_objective(self, X):
        y_predict = np.zeros(len(X))
        for i in range(len(X)):
            s = 0
            for alph, support_vectors_y, support_vectors in zip(self.alpha, self.support_vectors_y, self.support_vectors):
                s += alph * support_vectors_y * self.gaussian(X[i], support_vectors, self.gamma)
            y_predict[i] = s
        return y_predict

    def predict(self, X):
        X = np.atleast_2d(X)
        return np.sign(self.dual_objective(X))

    def linear(self, x1, x2):
        return np.dot(x1, x2)

    def gaussian(self, X, y, gamma):
        return np.exp(-(np.linalg.norm(X-y, ord=2)**2) / gamma)

if __name__ == "__main__":
    main()