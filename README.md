# This is a machine learning library developed by Caleb Johnson for CS5350/6350 in University of Utah

### To run the decision tree algorithm, you can use the run.sh script in the DecisionTree directory. The decision tree class has the follows methods to be used. 
### Decision Tree
* fit(X, y), takes training data and labels and computes information at each level to split the decision tree properly.
* score(X, y), takes in data and computes the accuracy score.
* predict(X), leveraged by score, but also takes data and predicts the label

### Adaboost

* fit(X, y, M=100), takes the training data and labels and runs the adaboost learning algorithm on it, iteratively updating results. M is an optional hyperparameter that gives the number of iterations.
* score(X, y),  takes in data and computes the accuracy score.
* calculate_error, calculate_alpha, and update_weights are all helper methods

### Bagging

I'll be honest, HW2 was a mess for me, currently bagging takes command line arguments for the training and test sets and computes accuracy

### Random Forest

Similar situation as above. It was a rough time for me, and I am just moving on from it and trying to do better going forward!

### LMS

The story continues

### Standard Perceptron
##### Parameters
* lr - learning rate
* epochs - number of epoches that the Perceptron will run for, default 10
##### Methods
* fit(X, y)	takes the training data and labels as input, updates weights and bias
* score(X, y)	take the instances and labels and returns an accuracy
* predict(X)	takes instances and provides predicted labels
* sign(value)	takes a value and transforms it to -1 and 1

### Voted Perceptron
##### Parameters
* Cms
* lr
* epochs
##### Methods
* fit(X, y)
* score(X, y)
* predict(X)
* sign(value)

### Averaged Perceptron
Parameters
* a
* lr
* epochs
Methods
* fit(X, y)
* score(X, y)
* predict(X)
* sign(value)

### PrimalSVM
Parameters
* epochs
* lr_update
* C
* bias
* weights
Methods
* fit(X, y)
* score(X, y)
* predict(X)

### DualSVM
Parameters
* C
* kernel
* gamma
* wstar
* bstar
* C
* gamma
* support_vectors
Methods
* fit(X, y)
* linear(X)
* gaussian(X, y, gamma)
* dual_objective(a, X, y)
* score(X, y)
* predict(X, kernel)

### KernelPerceptron
Parameters
* kernel
* T
* gamma
* alpha
Methods
* fit(X, y)
* dual_objective(X)
* predict(X)
* linear(X1, X2)
* gaussian(X, y, gamma)

