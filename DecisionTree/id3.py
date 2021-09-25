import sys
import pandas as pd
import math

def main():
	args = sys.argv[1:]
	if len(args) != 2:
		print("well thats just not ok")

	train = pd.read_csv(args[0])
	train_x = train.iloc[:, :-1]
	train_y = train.iloc[:, -1]

	classifications = train_y.unique()
	total_entropy = 0

	for classification in classifications:
		class_prop = (sum(train_y == classification)) / len(train_y)
		entropy = class_prop * math.log2(class_prop)
		total_entropy += entropy

	for classification in train_x.columns:
		average_entropy = 0
		print(classification)
		for classif in train_x[classification].unique():
			for label in classifications:
				count = sum(train_y[train_x[classification] == classif] == label)
				label_prop = count / len(train_y[train_x[classification] == classif])
				
				if label_prop > 0:
					average_entropy += (sum(train_x[classification] == classif) / len(train_x[classification])) * label_prop * math.log2(label_prop)
		print(average_entropy)
	print(-1*total_entropy)


def DecisionTree():
	def __init__(self):
		self.tree = None

	def fit(self, X, y, tree=None):
		self.tree = self.decisionTree(X, y)
		return self.tree

	def decisionTree(self, X, y):
		if X.empty:
			return
		node = self.calculateNode(X, y)

	def calculateNode(self, X, y):
		infoGains = []

if __name__ == "__main__":
    main()

