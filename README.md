# This is a machine learning library developed by Caleb Johnson for CS5350/6350 in University of Utah

### To run the decision tree algorithm, you can use the run.sh script in the DecisionTree directory. The decision tree class has the follows methods to be used. 
### Decision Tree
fit(X, y), takes training data and labels and computes information at each level to split the decision tree properly.
<br>
score(X, y), takes in data and computes the accuracy score.
<br>
predict(X), leveraged by score, but also takes data and predicts the label
<br>
### Adaboost

fit(X, y, M=100), takes the training data and labels and runs the adaboost learning algorithm on it, iteratively updating results. M is an optional hyperparameter that gives the number of iterations.
<br>
score(X, y),  takes in data and computes the accuracy score.
<br>
calculate_error, calculate_alpha, and update_weights are all helper methods
<br>
### Bagging

I'll be honest, HW2 was a mess for me, currently bagging takes command line arguments for the training and test sets and computes accuracy

### Random Forest

Similar situation as above. It was a rough time for me, and I am just moving on from it and trying to do better going forward!

### LMS

The story continues
