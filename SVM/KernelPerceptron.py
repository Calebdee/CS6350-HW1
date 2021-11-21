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

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.alpha = np.zeros(n_samples, dtype=np.float64)

        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                if self.kernel == "gaussian":
                    K[i,j] = self.gaussian(X[i], X[j], self.gamma)

        for t in range(self.T):
            for i in range(n_samples):
                if np.sign(np.sum(K[:,i] * self.alpha * y)) != y[i]:
                    self.alpha[i] += 1.0

        # Support vectors
        sv = self.alpha > 1e-5
        ind = np.arange(len(self.alpha))[sv]
        self.alpha = self.alpha[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]
        print("%d support vectors out of %d points" % (len(self.alpha),
                                                       n_samples))

    def project(self, X):
        y_predict = np.zeros(len(X))
        for i in range(len(X)):
            s = 0
            for a, sv_y, sv in zip(self.alpha, self.sv_y, self.sv):
                s += a * sv_y * self.gaussian(X[i], sv, self.gamma)
            y_predict[i] = s
        return y_predict

    def predict(self, X):
        X = np.atleast_2d(X)
        n_samples, n_features = X.shape
        #np.hstack((X, np.ones((n_samples, 1))))
        return np.sign(self.project(X))

    def linear_kernel(self, x1, x2):
        return np.dot(x1, x2)

    def polynomial_kernel(self, x, y, p=3):
        return (1 + np.dot(x, y)) ** p

    def gaussian(self, X, y, gamma):
        return np.exp(-(np.linalg.norm(X-y, ord=2)**2) / gamma)

if __name__ == "__main__":
    main()