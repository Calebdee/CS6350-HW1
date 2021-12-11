import numpy as np
import Layer
import sys
import pandas as pd

def main():
    args = sys.argv[1:]
    train = pd.read_csv(args[0], header=None)
    test = pd.read_csv(args[1], header=None)

    train = pd.read_csv(args[0], header=None)
    test = pd.read_csv(args[1], header=None)

    train_x = train.iloc[: , :-1]
    train_y = train.iloc[: , -1]
    test_x = test.iloc[: , :-1]
    test_y = test.iloc[: , -1]

    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x = np.array(test_x)

    weights = [5, 10, 25, 50, 100]
    gammas = [0.5, 0.5, 0.05, 0.1, 0.01]
    count = 0
    for weight in weights:
        test_y = np.array(test_y)

        net = NeuralNetwork(Layer.create_layers(3, 4, weight, 1))
        print("========================================")
        print("3 Layers - Width = " + str(weight))
        training_acc = net.fit(net, train_x, train_y, lr_0 = gammas[count], d = 1)
        print("training error: " + str(sum(training_acc)/len(training_acc)))
        testing_acc = net.score(net, test_x, test_y)
        count += 1


class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.epochs = 50

    def fit(self, net, X, y, lr_0 = 0.5, d = 1):
        all_losses = []

        for e in range(self.epochs):
            losses = []
            idxs = np.arange(len(X))

            # Reshuffle the data at the beginning of each epoch
            np.random.shuffle(idxs)
            for i in idxs:
                y_pred, zs = self.forward(X[i])
                losses.append(self.square_loss(y_pred, y[i]))

                lr = lr_0 / (1 + (lr_0/d)*e)
                self.backward(zs, y[i], lr)
            all_losses.append(np.mean(losses))

        return all_losses

    def score(self, net, X, y):
        losses = []
        for i in range(len(X)):
            y_pred, _ = self.forward(X[i])
            losses.append(self.square_loss(y_pred, y[i]))
        print("testing error:" + str(np.mean(losses)))

        return np.mean(losses)

    def forward(self, x): 
        x = np.append(1, x)
        zs = [np.atleast_2d(x)]

        for l in range(len(self.layers)):
            out = self.layers[l].evaluation(zs[l])
            zs.append(out)

        return float(zs[-1]), zs

    def backward(self, zs, y, lr = 0.1):

        partials = [zs[-1] - y]

        for l in range(len(zs) - 2, 0, -1):
            delta = self.layers[l].backwards(zs[l], partials)
            partials.append(delta)
    
        partials = partials[::-1]

        for l in range(len(self.layers)):
            grad = self.layers[l].update_ws(lr, zs[l], partials[l])


    def square_loss(self, pred, target):
        return 0.5*(pred - target)**2

if __name__ == "__main__":
    main()