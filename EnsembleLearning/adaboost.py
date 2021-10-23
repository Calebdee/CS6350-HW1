import sys
import pandas as pd
import math
import pprint
import numpy as np
from sklearn.preprocessing import LabelBinarizer, LabelEncoder

from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
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

# Define AdaBoost class
class AdaBoost:
    
    def __init__(self):
        self.alphas = []
        self.G_M = []
        self.M = None
        self.training_errors = []
        self.prediction_errors = []

    def fit(self, X, y, M = 100):
        self.alphas = [] 
        self.training_errors = []
        self.M = M
        y = y.to_numpy()

        # Iterate over M weak classifiers

        for m in range(0, M):
            
            # Set weights for current boosting iteration
            if m == 0:
                w_i = np.ones(len(y)) * 1 / len(y)  # At m = 0, weights are all the same and equal to 1 / N

            else:
                # (d) Update w_i
                w_i = self.update_weights(w_i, alpha_m, y, y_pred)

            
            
            # (a) Fit weak classifier and predict labels
            G_m = DecisionTreeClassifier(max_depth = 1)     # Stump: Two terminal-node classification tree
            G_m.fit(X, y, sample_weight = w_i)
            
            y_pred = G_m.predict(X)
            
            self.G_M.append(G_m) # Save to list of weak classifiers

            # (b) Compute error
            error_m = self.compute_error(y, y_pred, w_i)

            self.training_errors.append(error_m)

            # (c) Compute alpha
            alpha_m = self.compute_alpha(error_m)
            self.alphas.append(alpha_m)
    def score(self, X, y):
        model = self.G_M[self.M-1]
        print(model.score(X, y))

        

    # Compute error rate, alpha and w
    def compute_error(self, y, y_pred, w_i):
        sum = 0
        for i in range(len(y)):
        	if y[i] != y_pred[i]:
        		sum += w_i[i]
        return sum

    def compute_alpha(self, error):
        '''
        Calculate the weight of a weak classifier m in the majority vote of the final classifier. This is called
        alpha in chapter 10.1 of The Elements of Statistical Learning. Arguments:
        error: error rate from weak classifier m
        '''
        return 0.5 * np.log((1 - error) / error)

    def update_weights(self, w_i, alpha, y, y_pred):
        ''' 
        Update individual weights w_i after a boosting iteration. Arguments:
        w_i: individual weights for each observation
        y: actual target value
        y_pred: predicted value by weak classifier  
        alpha: weight of weak classifier used to estimate y_pred
        '''  
        wt_i = []
        for i in range(len(y)):
        	w_i[i] += w_i[i]  * math.exp(-1*alpha * (y[i] * y_pred[i]))
        w_i = (w_i / sum(w_i))
        return w_i

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
        self.tree = self.decisionTree(X, y, depth=0)
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
            if self.learnType == "majorityError":
                infoGains.append(self.majorityError(X, y)-self.attribute_majorityError(X, y,key))
        
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

    def majorityError(self, X, y):
        Class = y.keys()
        majority = y[self.labelName].value_counts().argmax()

        prop = y[self.labelName].value_counts()[majority]/len(y[self.labelName])

        return (1 - prop)

    def attribute_majorityError(self, X, y,attribute):
        Class = y[self.labelName].keys()
        uniqueOutputs = y[self.labelName].unique()
        variables = X[attribute].unique()   
        totalError = 0
        
        for variable in variables:
            majorityProp = 0
            for target_variable in uniqueOutputs:
                num = len(X[attribute][X[attribute]==variable][y[self.labelName] == target_variable])
                den = len(X[attribute][X[attribute]==variable])
                fraction = num/(den+self.eps)
                if fraction > majorityProp:
                    majorityProp = fraction
            fraction2 = len(X[attribute][X[attribute]==variable]) / len(y)

            totalError += -fraction2*(1 - majorityProp)
        return abs(totalError)
    
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

