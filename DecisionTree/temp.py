import sys
import pandas as pd
import math
import pprint
import numpy as np
from numpy import log2 as log

def main():
    args = sys.argv[1:]
    if len(args) != 2:
        print("well thats just not ok")

    train = pd.read_csv(args[0], header=None)
    test = pd.read_csv(args[1], header=None)
    test = np.array(test)
    train_test = np.array(train)
    train_x = pd.DataFrame(train.iloc[:, :-1])
    train_y = pd.DataFrame(train.iloc[:, -1])
    types = ["entropy", "gini"]

    print("---------------------------------")
    print("| Training ID3 Algorithm Table  |")
    print("---------------------------------")
    print("| Type    | Max Depth | Score   |")
    print("---------------------------------")
    for i in range(1, 7):
        for version in types:
            tree = DTClassifier(i, train.columns[-1], version)
            mytree = tree.fit(train_x, train_y)
            if version == "gini":
                version += "   "
            acc = tree.score(train_test)
            print("| " + version + " | " + str(i) + "         | " + "{:.4f}".format(acc) + "  |")

    print("---------------------------------")
    print("")

    print("---------------------------------")
    print("| Testing ID3 Algorithm Table   |")
    print("---------------------------------")
    print("| Type    | Max Depth | Score   |")
    print("---------------------------------")
    for i in range(1, 7):
        for version in types:
            tree = DTClassifier(i, train.columns[-1], version)
            mytree = tree.fit(train_x, train_y)
            if version == "gini":
                version += "   "
            acc = tree.score(test)
            print("| " + version + " | " + str(i) + "         | " + "{:.4f}".format(acc) + "  |")
    print("---------------------------------")
    

    
    
        

class DTClassifier():
    def __init__(self,maxDepth, labeLName, learnType):
        self.eps = np.finfo(float).eps
        self.tree = None
        self.maxDepth = maxDepth
        self.labelName = labeLName
        self.learnType = learnType
        self.shiftInfoGain = []
        self.modeClassification = None
        

    def fit(self, X, y, tree=None):
        self.modeClassification = y[self.labelName].mode()
        self.tree = self.decisionTree(X, y, depth=1)
        return self.tree
    
    def decisionTree(self, X, y, depth, tree=None, lastPass=False):
        if depth > self.maxDepth:
            return
        if X.empty:
            return
        Class = y.keys()
    
        node = self.calculateNode(X, y)

        availableAttributes = np.unique(X[node])
        
        # Create dictionary at each level of the tree
        if tree is None:                    
            tree={}
            tree[node] = {}

        for value in availableAttributes:

            branch_x, branch_y = self.get_subsets(X,y,node,value)
            availableOutputs,counts = np.unique(y,return_counts=True)                        

            if len(counts)==1:#Checking purity of subset
                tree[node][value] = availableOutputs[0]                                                    
            else:        
                if len(branch_x.columns) == 1 and lastPass == False:
                    tree[node][value] = self.decisionTree(branch_x, branch_y, depth+1, None, True)
                else:
                    tree[node][value] = self.decisionTree(branch_x.drop(node, axis=1), branch_y, depth+1)

        return tree
    
    def calculateNode(self, X, y):
        infoGains = []
        for key in X.keys():
            if self.learnType == "entropy":
                infoGains.append(self.entropy(X, y)-self.attribute_entropy(X, y,key))
            if self.learnType == "gini":
                infoGains.append(self.gini(X, y)-self.attribute_gini(X, y,key))
                #print(infoGains)
            if self.learnType == "me":
                print("here")
            
        
        if infoGains[np.argmax(infoGains)] > 0:
            self.shiftInfoGain.append(infoGains[np.argmax(infoGains)])
        return X.keys()[np.argmax(infoGains)]
    
    def get_subsets(self, X, y, node,value):
        return X[X[node] == value], y[X[node] == value]
    
    def entropy(self, X, y):
        Class = y.keys()
        entropy = 0
        values = y[self.labelName].unique()
        for value in values:
            fraction = y[self.labelName].value_counts()[value]/len(y[self.labelName])
            entropy += -fraction*np.log2(fraction)
 
        return entropy

    def gini(self, X, y):
        Class = y.keys()
        gini = 0
        values = y[self.labelName].unique()
        for value in values:
            fraction = y[self.labelName].value_counts()[value]/len(y[self.labelName])
            gini += fraction*fraction
 
        return (1 - gini)
    
    def attribute_entropy(self, X, y,attribute):
        Class = y[self.labelName].keys()
        uniqueOutputs = y[self.labelName].unique()
        variables = X[attribute].unique()   
        totalEntropy = 0
        
        for variable in variables:
            entropy = 0
            for target_variable in uniqueOutputs:
                num = len(X[attribute][X[attribute]==variable][y[self.labelName] == target_variable])
                den = len(X[attribute][X[attribute]==variable])
                fraction = num/(den+self.eps)
                entropy += -fraction*log(fraction+self.eps)
            fraction2 = den/len(y)
            totalEntropy += -fraction2*entropy
        return abs(totalEntropy)

    def attribute_gini(self, X, y,attribute):
        Class = y[self.labelName].keys()
        uniqueOutputs = y[self.labelName].unique()
        variables = X[attribute].unique()   
        totalGini = 0
        
        for variable in variables:
            gini = 0
            for target_variable in uniqueOutputs:
                num = len(X[attribute][X[attribute]==variable][y[self.labelName] == target_variable])
                den = len(X[attribute][X[attribute]==variable])
                fraction = num/(den+self.eps)
                gini += fraction*fraction
            fraction2 = den/len(y)
            totalGini += fraction2*gini
        return (1 - totalGini)

    def predict(self, X):
        """ Predict all classes for a dataset X

        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets

        Returns:
            array, shape (n_samples,)
                Predicted target values per element in X.
        """
        
        preds = []
        vals = []

        for row in range(0, X.shape[0]):
            preds.append(self.findPrediction(X[row,:-1], self.tree))
            vals.append(X[row,-1])
        return preds, vals
    
    def findPrediction(self, X, tree):
        if isinstance(tree, dict):
            for key in tree.keys():
                if X[key] in tree[key]:
                    return self.findPrediction(X, tree[key][X[key]])
                else:
                    return self.modeClassification[0]
        else:
            return tree


    def score(self, data):
        """ Return accuracy(Classification Acc) of model on a given dataset. Must implement own score function.

        Args:
            X (array-like): A 2D numpy array with data, excluding targets
            y (array-like): A 1D numpy array of the targets 
        """
        preds, vals = self.predict(data)
        num_correct = 0
        total = 0
        for i in range(len(preds)):
            if preds[i] == vals[i]:
                num_correct += 1
            total += 1

        return (num_correct/total)
    
    def getShiftInfoGain(self):
        return self.shiftInfoGain

if __name__ == "__main__":
    main()

