import sys
import pandas as pd
import math
import pprint
import numpy as np

from sklearn import preprocessing 
from id3 import DTClassifier
from collections import Counter
from matplotlib import pyplot as plt

def main():
	args = sys.argv[1:]
	if len(args) != 2:
		print("well thats just not ok")
	train = pd.read_csv(args[0], header=None)
	test = pd.read_csv(args[1], header=None)
	y = pd.DataFrame(test.iloc[:, -1]).to_numpy()
	t_y = pd.DataFrame(train.iloc[:, -1]).to_numpy()
	test = np.array(test)
	train_test = np.array(train)

	y_preds = {}
	t_y_preds = {}

	training_acc = []
	testing_acc = []

	for j in range(2, 501):
		for i in range(1, j):
			train_sample = train.sample(frac=1.0, replace=True).reset_index(drop=True)
			train_x = pd.DataFrame(train_sample.iloc[:, :-1])
			train_y = pd.DataFrame(train_sample.iloc[:, -1])

			tree = DTClassifier(100, train_sample.columns[-1], "entropy")
			mytree = tree.fit(train_x, train_y)

			preds = tree.predict(test)[0]
			t_preds = tree.predict(train_test)[0]
			for i in range(len(preds)):
				if i in y_preds.keys():
					y_preds[i].append(preds[i])
				else:
					y_preds[i] = []
					y_preds[i].append(preds[i])
			for i in range(len(t_preds)):
				if i in t_y_preds.keys():
					t_y_preds[i].append(t_preds[i])
				else:
					t_y_preds[i] = []
					t_y_preds[i].append(t_preds[i])

		train_correct = 0
		test_correct = 0
		for i in range(test.shape[0]):
			if most_common(y_preds.get(i)) == y[i]:
				test_correct += 1
		for i in range(train.shape[0]):
			if most_common(t_y_preds.get(i)) == t_y[i]:
				train_correct += 1

		training_acc.append(train_correct / train.shape[0])
		training_acc.append(test_correct / test.shape[0])
		y_preds.clear()
		t_y_preds.clear()
	plt.plot(list(range(1, 501)), training_acc)
	plt.plot(list(range(1, 501)), testing_acc)
	plt.xlabel('Number of Trees')
	plt.ylabel('Accuracy')
	plt.title('Bagging Training and Test Accuracies')
	plt.legend(["Train",  "Test"])
	plt.show()

def most_common(lst):
    return max(set(lst), key=lst.count)

if __name__ == "__main__":
    main()