import sys
import pandas as pd
import math
from sklearn.utils import shuffle

def main():
	args = sys.argv[1:]
	train = pd.read_csv(args[0], header=None)
	test = pd.read_csv(args[1], header=None)

	train_x = train.iloc[: , :-1]
	train_y = train.iloc[: , -1]

	test_x = test.iloc[: , :-1]
	test_y = test.iloc[: , -1]

	print(train_y)

def PrimalSVM():
	def __init__(self, lr, epochs=10):
		self.epochs = epochs

	def fit(self, X, y):
		self.weights = pd.Series([0]*(len(X.columns)+1))
		for i in range(self.epochs):
			X = shuffle(X)
			print("EPOCH " + str(i+1))

	def score(self, X, y):
		self.predict(X)

	def predict(self, X):
		print("predict")

if __name__ == "__main__":
    main()