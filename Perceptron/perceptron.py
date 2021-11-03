import sys
import pandas as pd
import math

import numpy as np
from numpy import log2 as log
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype

from sklearn.utils import shuffle

def main():
	args = sys.argv[1:]
	train = pd.read_csv(args[0], header=None)
	test = pd.read_csv(args[1], header=None)

	train_x = train.iloc[: , :-1]
	train_y = train.iloc[: , -1]

	test_x = test.iloc[: , :-1]
	test_y = test.iloc[: , -1]

	percep = Perceptron(lr=0.1, epochs=10)
	percep.fit(train_x, train_y)
	print("\nLearned Weight Vector")
	print(percep.weights.to_string())
	print("\nTraning Error")
	print(1 - percep.score(train_x, train_y))
	print("Testing Error")
	print(1 - percep.score(test_x, test_y))
	print("\n")

class Perceptron():
	def __init__(self, lr, epochs=10):
		self.weights = []
		self.lr = lr
		self.epochs = epochs

	def fit(self, X, y):
		self.weights = pd.Series([0]*(len(X.columns)+1))

		for i in range(self.epochs):
			print("EPOCH " + str(i+1))
			# We want to shuffle the rows of the dataset at each pass-through
			X = shuffle(X)
			for index, row in X.iterrows():
				row = pd.Series([1]).append(row, ignore_index=True)

				y_prime = self.sign(self.weights.dot(row))
				if self.sign(y[index]) != y_prime:
					self.weights = self.weights + (self.lr * (self.sign(y[index])*row))


	def score(self, X, y):
		correct = 0
		length = len(y)
		preds = self.predict(X)

		for i in range(length):
			if preds[i] == self.sign(y[i]):
				correct += 1

		return correct/length

	def predict(self, X):
		preds = []
		for index, row in X.iterrows():
			row = pd.Series([1]).append(row, ignore_index=True)
			preds.append(self.sign(self.weights.dot(row)))
		return pd.Series(preds)

	def sign(self, value):
		if value <= 0:
			return -1
		else:
			return 1

if __name__ == "__main__":
    main()