import sys
import pandas as pd
import math
import numpy as np
from scipy.optimize import minimize

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

	gamma_0 = 1
	alpha = 1

	lr_update1 = lambda t : gamma_0 / (1 + (gamma_0/alpha)*t)
	lr_update2 = lambda t : gamma_0 / (1 + t)
	Cs = [500/873]
	gamma = [0.1, 0.5, 1.0, 5.0, 100.0]
	# print("========================================================")
	# print("| Dual Domain - SsGD - Linear                          |")
	# print("========================================================")
	# for c in Cs:
	# 	# gamm in gamma:
	# 	svm = DualSVM(train_x, train_y, C=c, kernel="linear", gamma=None)
	# 	svm.fit(train_x, train_y)
	# 	print("| C with a value of " + str(round(c*873)) + "/873                            |")
	# 	print("| Learned Weights: " + str(svm.wstar) + "                            |")
	# 	print("| Learned Bias: " + str(svm.bstar) + "                                       |")
	# 	print("| Learned Support Trees: " + str(len(svm.support_vectors)) + "                   |")
	# 	print("| Training Accuracy: %.3f" % svm.score(train_x, train_y) + "                             |")
	# 	print("| Testing Accuracy: %.3f" % svm.score(test_x, test_y) + "                              |")
	#	print("========================================================")

	svs = []
	print("========================================================")
	print("| Dual Domain - SsGD - Gaussian Kernel                 |")
	print("========================================================")
	for c in Cs:
		for gamm in gamma:
			svm = DualSVM(train_x, train_y, C=c, kernel="gaussian", gamma=gamm)
			svm.fit(train_x, train_y)
			print("| C with a value of " + str(round(c*873)) + "/873 and gamma %d              |" % gamm)
			print("| Learned Weights: " + str(svm.wstar) + "                            |")
			print("| Learned Bias: " + str(svm.bstar) + "                                       |")
			print("| Learned Support Trees: " + str(len(svm.support_vectors)) + "                   |")
			print("| Training Accuracy: %.3f" % svm.score(train_x, train_y) + "                             |")
			print("| Testing Accuracy: %.3f" % svm.score(test_x, test_y) + "                              |")
			print("========================================================")
			if c == 500/873:
				svs.append(svm.support_vectors)

	for i in range(len(svs)):
		count = 0
		for v in np.array(svs[i]):
			if v in np.array(sv[i+1]):
				count += 1

		print("Overlap " + str(gamma[i]) + " to " + str(gamma[i+1]) + ": " + str(count))

class DualSVM:
    def __init__(self, X, y, C, kernel = "dot", gamma=0.1):
        self.wstar = np.ndarray
        self.bstar = float
        self.C = C
        self.gamma = gamma
        self.support_vectors = []
        self.kernel = kernel

    def fit(self, X, y):
        constraints = [
            {
                'type': 'eq',
                'fun': lambda a: np.sum(a*y)
            },
            {
                'type': 'ineq',
                'fun': lambda a : a 
            },
            {
                'type': 'ineq',
                'fun': lambda a: self.C - a 
            }
        ]
        
        solution = minimize(self.dual_objective, x0=np.zeros(shape=(len(X),)), args=(X, y), method='SLSQP', constraints=constraints)

        self.support_vectors = np.where(0 < solution.x)[0]  

        self.wstar = np.zeros_like(X[0])
        for i in range(len(X)):
            self.wstar += solution['x'][i]*y[i]*X[i]

        self.bstar = 0
        for j in range(len(X)):
            self.bstar += y[j] - np.dot(self.wstar, X[j])
        self.bstar /= len(X)

    def linear(self, X):
        return (X@X.T)

    def gaussian(self, X, y, gamma):
        return exp(-(np.linalg.norm(x-y, ord=2)**2) / gamma)

    def dual_objective(self, a, X, y):
        decisionMatrix = y * np.ones((len(y), len(y)))
        alphaMatrix = a * np.ones((len(a), len(a)))

        if self.kernel == 'linear':
            xis = self.linear(X)
        if self.kernel == 'gaussian':
            xis = X**2 @ np.ones_like(X.T) - 2*X@X.T + np.ones_like(X) @ X.T**2 
            xis = np.exp(-( xis / self.gamma))

        vals = (decisionMatrix*decisionMatrix.T) * (alphaMatrix*alphaMatrix.T) * xis
        return 0.5*np.sum(vals) - np.sum(a)

    def score(self, X, y):
    	correct = 0
    	length = len(y)
    	preds = self.predict(X)

    	for i in range(length):
    		if preds[i] == y[i]:
    			correct += 1
    	return correct/length

    def predict(self, X, kernel = "linear") -> np.ndarray:
        if kernel == 'linear':
            pred = lambda d : np.sign(np.dot(self.wstar, d) + self.bstar)
        if kernel == 'gaussian':
            pred = lambda d : np.sign(self.gaussian(self.wstar, d, self.gamma) + self.bstar)
        return np.array([pred(xi) for xi in X])



if __name__ == "__main__":
    main()