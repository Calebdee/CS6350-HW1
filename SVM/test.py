import numpy as np
X = np.array([[1, 0.5, -1, 0.3], [1, -1, -2, -2], [1, 1.5, 0.2, -2.5]])
y = np.array([1, -1, 1])
C = 1/3
weights = np.array([0,0,0,0])
lr = [0.01, 0.005, 0.0025]

Xi = np.arange(len(X))
#np.random.shuffle(Xi)

for epoch in range(len(lr)):
	print("Epoch: %d" % epoch)
	for index in Xi:
		print(np.sign(y[index]) * np.dot(weights, X[index]))
		if np.sign(y[index]) * np.dot(weights, X[index]) <= 1:
			print("update under")
			nweights = weights.copy()
			nweights[0] = 0
			weights = weights - lr[epoch]*nweights + lr[epoch]*C*len(Xi)*np.sign(y[index])*X[index]

		else:
			print("update over")
			weights = (1-lr[epoch])*weights

		print(weights)

print("FINAL")
print(weights)