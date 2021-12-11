import numpy as np

def create_layers(num, into, hidden, out):
    layers = []
    layers.append(ModelLayer(in_channels = into, out_channels = hidden, activation = 'sigmoid', weight_init='zeroes'))
    for i in range(num-1):
        layers.append(ModelLayer(in_channels = hidden, out_channels = hidden, activation = 'sigmoid', weight_init='zeroes'))
    layers.append(ModelLayer(in_channels=hidden, out_channels=out, activation = 'identity', weight_init='zeroes', include_bias=False))
    return layers

class ModelLayer:
    def sigmoid(self, x):
        sigma = 1 / (1 + np.exp(-x))
        return sigma

    def sigmoid_prime(self, x):
        sigma = 1 / (1 + np.exp(-x))
        return sigma * (1 - sigma)

    def identity(self, x):
        return x

    def identity_prime(self, x):
        return 1

    def __init__(self, in_channels, out_channels, activation, weight_init, include_bias = True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation

        if include_bias:
            shape = (self.in_channels+1, self.out_channels+1)
        else:
            shape = (self.in_channels+1, self.out_channels)

        if weight_init == 'zeroes':
            self.layer_weights = np.zeros(shape, dtype=np.float128)
        elif weight_init == 'random':
            self.layer_weights = np.random.standard_normal(shape)
        else: raise NotImplementedError
            
    def __str__(self) -> str:
        return str(self.layer_weights)
    
    def evaluation(self, x):
        if self.activation == 'sigmoid':
            return self.sigmoid(np.dot(x, self.layer_weights))
        else:
            return np.dot(x, self.layer_weights)
    
    def backwards(self, zs, partials):
        delta = np.dot(partials[-1], self.layer_weights.T)
        if self.activation == "sigmoid":
            delta *= self.sigmoid_prime(zs)
            return delta
        else:
            return delta

    
    def update_ws(self, lr, zs, partials):
        grad = zs.T.dot(partials)
        self.layer_weights += -lr * grad
        return grad