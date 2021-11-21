import sys
import pandas as pd
import math
import numpy as np

def main():
	args = sys.argv[1:]
	train = pd.read_csv(args[0], header=None)
	test = pd.read_csv(args[1], header=None)

	train_x = train.iloc[: , :-1]
	train_y = train.iloc[: , -1]

	test_x = test.iloc[: , :-1]
	test_y = test.iloc[: , -1]

	train_x = np.array(train_x)
	train_y = np.array(train_y)
	test_x = np.array(test_x)
	test_y = np.array(test_y)

	gamma_0 = 1
	alpha = 1

	lr_update1 = lambda t : gamma_0 / (1 + (gamma_0/alpha)*t)
	lr_update2 = lambda t : gamma_0 / (1 + t)
	Cs = [100/873, 500/873, 700/873] 
	print("======================================================================================")
	print("| Primal Domain - Stochastic Subgradient Descent                                     |")
	print("| Learning Update of gamma_0 / (1 + (gamma_0/alpha)*t)                               |")
	print("======================================================================================")
	for c in Cs:
		svm = PrimalSVM(C=c, lr_update=lr_update1, epochs=100)
		svm.fit(train_x, train_y)
		print("| C with a value of " + str(round(c*873)) + "/873                                                          |")
		print("| Trained Weights/Bias: " + str(svm.weights) + "|")
		print("| Training Accuracy: %.3f" % svm.score(train_x, train_y) + "                                                           |")
		print("| Testing Accuracy: %.3f" % svm.score(test_x, test_y) + "                                                            |")
		print("======================================================================================")

	print("======================================================================================")
	print("| Primal Domain - Stochastic Subgradient Descent                                     |")
	print("| Learning Update of gamma_0 / (1 + t)                                               |")
	print("======================================================================================")
	for c in Cs:
		svm = PrimalSVM(C=c, lr_update=lr_update2, epochs=100)
		svm.fit(train_x, train_y)
		print("| C with a value of " + str(round(c*873)) + "/873                                                          |")
		print("| Trained Weights/Bias: " + str(svm.weights) + "|")
		print("| Training Accuracy: %.3f" % svm.score(train_x, train_y) + "                                                           |")
		print("| Testing Accuracy: %.3f" % svm.score(test_x, test_y) + "                                                            |")
		print("======================================================================================")


class PrimalSVM():
	def __init__(self, C, lr_update, epochs=10):
		self.epochs = epochs
		self.lr_update = lr_update
		self.C = C
		self.bias = 0
		self.weights = 0

	def fit(self, X, y):
		X = np.insert(X, 0, [1]*len(X), axis=1)
		self.weights = np.zeros_like(X[0])

		for t in range(self.epochs):
			Xi = np.arange(len(X))
			np.random.shuffle(Xi)
			lr = self.lr_update(t)

			for index in Xi:
				if self.sign(y[index]) * np.dot(self.weights, X[index]) <= 1:
					nweights = self.weights.copy()
					nweights[0] = 0
					self.weights = self.weights - lr*nweights + lr*self.C*len(Xi)*self.sign(y[index])*X[index]
				else:
					self.weights = (1-lr)*self.weights

	def score(self, X, y):
		correct = 0
		length = len(y)
		preds = self.predict(X)

		for i in range(length):
			if preds[i] == self.sign(y[i]):
				correct += 1
		return correct/length

	def predict(self, X) -> np.ndarray:
		X = np.insert(X, 0, [1]*len(X), axis=1)
		pred = lambda d : np.sign(np.dot(self.weights, d))
		return np.array([pred(xi) for xi in X])

	def sign(self, value):
		if value <= 0:
			return -1
		else:
			return 1

if __name__ == "__main__":
    main()