"""
implement the class specifications for a basic MLP, optimizer, .
Follow the instructions provided in the writeup to completely
You will be able to test each section individually by submitting
to autolab after implementing what is required for that section
-- do not worry if some methods required are not implemented yet.

Notes:

The __call__ method is a special reserved method in
python that defines the behaviour of an object when it is
used as a function. For example, take the Linear activation
function whose implementation has been provided.

# >>> activation = Identity()
# >>> activation(3)
# 3
# >>> activation.forward(3)
# 3
"""

# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)
import numpy as np
import os


class Activation(object):

    """
    Interface for activation functions (non-linearities).

    In all implementations, the state attribute must contain the result, i.e. the output of forward (it will be tested).
    """

    # No additional work is needed for this class, as it acts like an abstract base class for the others

    def __init__(self):
        self.state = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented


class Identity(Activation):

    """
    Identity function (already implemented).
    """

    # This class is a gimme as it is already implemented for you as an example

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        self.state = x
        return x

    def derivative(self):
        return 1.0


class Sigmoid(Activation):

    """
    Sigmoid non-linearity
    """

    # Remember do not change the function signatures as those are needed to stay the same for AL

    def __init__(self):
        super(Sigmoid, self).__init__()
    
    def _sigmoid(self, x):
        return 1. / (1. + np.exp(-x))

    def forward(self, x):
        self.state = self._sigmoid(x)
        return self.state

    def derivative(self):
        return self.state * (1. - self.state)


class Tanh(Activation):

    """
    Tanh non-linearity
    """

    # This one's all you!

    def __init__(self):
        super(Tanh, self).__init__()

    def _tanh(self, x):
        return np.tanh(x)

    def forward(self, x):
        self.state = self._tanh(x)
        return self.state

    def derivative(self):
        return 1 - self.state**2


class ReLU(Activation):

    """
    ReLU non-linearity
    """

    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        self.state = x
        x[x<0] = 0
        return x

    def derivative(self):
        d = np.zeros(self.state.shape)
        d[self.state > 0] = 1
        return d

# Ok now things get decidedly more interesting. The following Criterion class
# will be used again as the basis for a number of loss functions (which are in the
# form of classes so that they can be exchanged easily (it's how PyTorch and other
# ML libraries do it))


class Criterion(object):

    """
    Interface for loss functions.
    """

    # Nothing needs done to this class, it's used by the following Criterion classes

    def __init__(self):
        self.logits = None
        self.labels = None
        self.loss = None

    def __call__(self, x, y):
        return self.forward(x, y)

    def forward(self, x, y):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented


class SoftmaxCrossEntropy(Criterion):

    """
    Softmax loss
    """

    def __init__(self):

        super(SoftmaxCrossEntropy, self).__init__()
        self.sm = None

    def forward(self, x, y):

        self.logits = x
        self.labels = y

        k = y.shape[1]
        self.sm = np.exp(x) / np.repeat(np.sum(np.exp(x), axis=1), k).reshape(-1, k)

        log_likelihood = -np.log(self.sm[y == 1.])
        return log_likelihood

    def derivative(self):
        return self.sm - self.labels


class BatchNorm(object):

    def __init__(self, fan_in, alpha=0.9):

        # You shouldn't need to edit anything in init

        self.alpha = alpha
        self.eps = 1e-8
        self.x = None
        self.norm = None
        self.out = None

        # The following attributes will be tested
        self.var = np.ones((1, fan_in))
        self.mean = np.zeros((1, fan_in))

        self.gamma = np.ones((1, fan_in))
        self.dgamma = np.zeros((1, fan_in))

        self.beta = np.zeros((1, fan_in))
        self.dbeta = np.zeros((1, fan_in))

        # inference parameters
        self.running_mean = np.zeros((1, fan_in))
        self.running_var = np.ones((1, fan_in))

    def __call__(self, x, eval=False):
        return self.forward(x, eval)

    def forward(self, x, eval=False):

        self.x = x

        # print('mean bef: ', self.mean.shape)
        # self.mean = np.mean(x, axis=0)
        # print('mean : ', self.mean.shape)

        # print('var bef: ', self.var.shape)
        # self.var = np.var(x, axis=0)
        # print('var : ', self.var.shape)

        # self.norm = (x - self.mean) / np.sqrt(self.var + self.eps)
        # print('norm : ', self.norm.shape)
        # self.out = self.gamma * self.norm + self.beta
        # print('out : ', self.out.shape)

        # update running batch statistics
        # self.running_mean = # ???
        # self.running_var = # ???

        if not eval:
            # self.mean = np.mean(x, axis=0).reshape(1, -1)
            # self.var = np.var(x, axis=0).reshape(1, -1)
            # print('mean bef: ', self.mean.shape)
            # print('var bef: ', self.var.shape)    
            self.mean = np.mean(x, axis=0)
            self.var = np.var(x, axis=0)
            self.norm = (x - self.mean) / np.sqrt(self.var + self.eps)
            self.out = self.gamma*self.norm + self.beta

            self.running_mean = self.alpha * self.running_mean + (1 - self.alpha) * self.mean
            self.running_var = self.alpha * self.running_var + (1 - self.alpha) * self.var
            self.cache = (x, self.norm, self.mean, self.var, self.gamma, self.beta)

            # print('x : ', x.shape)
            
            # print('mean : ', self.mean.shape)
            # print('var : ', self.var.shape)
            # print('norm : ', self.norm.shape)
            # print('out : ', self.out.shape)
        else:
            self.norm = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            self.out = self.gamma*self.norm + self.beta
    
        return self.out
    
    def backward(self, delta):

        X, X_norm, mu, var, gamma, beta = self.cache

        N = X.shape[0]

        # print('x : ', x.shape)
            
        # print('mean : ', self.mean.shape)
        # print('var : ', self.var.shape)
        # print('norm : ', self.norm.shape)
        

        # X_mu = X - self.mean
        # std_inv = 1. / np.sqrt(self.var + self.eps)

        # dX_norm = delta * self.gamma
        # dvar = np.sum(dX_norm * X_mu, axis=0) * -.5 * std_inv**3
        # dmu = np.sum(dX_norm * -std_inv, axis=0) + dvar * np.mean(-2. * X_mu, axis=0)

        # dX = (dX_norm * std_inv) + (dvar * 2 * X_mu / N) + (dmu / N)
        
        self.dgamma = np.sum(delta * self.norm, axis=0)
        self.dbeta = np.sum(delta, axis=0)

        # print('dgamma : ', self.dgamma.shape)
        # print('dbeta : ', self.dbeta.shape)

        dX = (1. / N) * self.gamma / np.sqrt(self.var + self.eps) * (N * delta - self.dbeta - self.dgamma * self.norm)

        # print('dX : ', dX.shape)
        #unfold the variables stored in cache
        # xhat,gamma,xmu,ivar,sqrtvar,var,eps = cache

        #get the dimensions of the input/output
        # N, D = delta.shape

        #step9
        # self.dbeta = np.sum(delta, axis=0).reshape(1, -1)
        # self.dgamma = np.sum(delta*self.norm, axis=0).reshape(1, -1)
        # dNorm = delta*self.gamma
        # print('dNorm : ', dNorm.shape)
        # dVar = np.sum(dNorm * (self.x - self.mean) * (-1./2) * (self.var + self.eps)**(-1.5), axis=0)
        # print('dVar : ', dVar.shape)
        # dMean = np.sum(dNorm * (-1.) * (self.var + self.eps)**(-0.5), axis=0) + dVar * np.sum(-2 * (self.x - self.mean)) * (1. / N)
        # print('dMean : ', dMean.shape)
        # dX = dNorm * (self.var + self.eps)**(-0.5) + dVar * 2 * (self.x - self.mean) * (1. / N) + dMean * (1. / N)
        # dX = (1. / N) * self.gamma / np.sqrt(self.var + self.eps) * (N * delta - np.sum(delta, axis=0)
            # - (self.x - self.mean) * (self.var + self.eps)**(-1.0) * np.sum(delta * (self.x - self.mean), axis=0))

        return dX


# These are both easy one-liners, don't over-think them
def random_normal_weight_init(d0, d1):
    return np.random.randn(d0, d1)


def zeros_bias_init(d):
    return np.zeros((1, d))


class MLP(object):

    """
    A simple multilayer perceptron
    """

    def __init__(self, input_size, output_size, hiddens, activations, weight_init_fn, bias_init_fn, criterion, lr, momentum=0.0, num_bn_layers=0):
        np.set_printoptions(precision=8, edgeitems=3)
        # Don't change this -->
        self.train_mode = True
        self.num_bn_layers = num_bn_layers
        self.bn = num_bn_layers > 0
        self.nlayers = len(hiddens) + 1
        self.input_size = input_size
        self.output_size = output_size
        self.activations = activations
        self.criterion = criterion
        self.lr = lr
        self.momentum = momentum
        # <---------------------

        # Don't change the name of the following class attributes,
        # the autograder will check against these attributes. But you will need to change
        # the values in order to initialize them correctly
        layers = [input_size] + hiddens + [output_size]
        self.W = [weight_init_fn(layers[i], layers[i+1]) for i in range(len(layers) - 1)]
        self.dW = [np.zeros(self.W[i].shape) for i in range(len(self.W))]
        self.b = [bias_init_fn(layers[i]) for i in range(1, len(layers))]
        self.db = [np.zeros(self.b[i].shape) for i in range(len(self.b))]

        # if batch norm, add batch norm parameters
        if self.bn:
            self.bn_layers = [BatchNorm(layers[i+1]) for i in range(num_bn_layers)]
        else:
            self.bn_layers = None

        self.outputs = []
        for i in range(len(layers)):
            self.outputs.append(np.zeros([layers[i]]))

        self.v_w = [np.zeros(dW.shape) for dW in self.dW]
        self.v_b = [np.zeros(db.shape) for db in self.db] 

        # Feel free to add any other attributes useful to your implementation (input, output, ...)

    def forward(self, x):
        self.outputs[0] = x
        for i, act in enumerate(self.activations):
            x_inp = self.outputs[i]
            # print('W[i]: ', self.W[i].shape)
            # print('b[i]: ', self.b[i].shape)
            # print('outputs[i]: ', x_inp.shape)
            
            # print('z: ', z.shape)
            if i < self.num_bn_layers:
                z = np.dot(x_inp, self.W[i]) + self.b[i]
                # print("before batch_norm: ", z, z.shape)
                z = self.bn_layers[i].forward(z, not self.train_mode)
                # print("after batch_norm: ", z, z.shape)
                self.outputs[i+1] = act.forward(z)
            else:
                z = np.dot(x_inp, self.W[i]) + self.b[i]
                self.outputs[i+1] = act.forward(z)

        return self.outputs[-1]

    def zero_grads(self):
        self.dW = [np.zeros(dW.shape) for dW in self.dW]
        self.db = [np.zeros(db.shape) for db in self.db]
        if self.bn_layers:
            for j in range(len(self.bn_layers)):
                self.bn_layers[j].dgamma = np.zeros(self.bn_layers[j].dgamma.shape)
                self.bn_layers[j].dbeta = np.zeros(self.bn_layers[j].dbeta.shape)

    def step(self):
        # Momentum update
        # v = mu * v - learning_rate * dx # integrate velocity
        # x += v # integrate position

        #Perform weight and bias updates
        for j in range(len(self.W)):
            self.v_w[j] = self.momentum * self.v_w[j] - self.lr * self.dW[j]
            self.v_b[j] = self.momentum * self.v_b[j] - self.lr * self.db[j]
            self.W[j] = self.W[j] + self.v_w[j]
            self.b[j] = self.b[j] + self.v_b[j]
        
        if self.bn_layers:
            for j in range(len(self.bn_layers)):
                self.bn_layers[j].gamma -= self.lr * self.bn_layers[j].dgamma
                self.bn_layers[j].beta -= self.lr * self.bn_layers[j].dbeta

    def backward(self, labels):
        _ = self.criterion.forward(self.outputs[-1], labels)
        deriv = self.criterion.derivative()

        batch_size = self.outputs[-1].shape[0]
        dErr = self.activations[-1].derivative() * deriv
        
        self.dW[-1] = np.dot(self.outputs[-2].T, dErr) / batch_size
        self.db[-1] = np.dot(np.ones((self.outputs[-2].shape[0], )).T, dErr) / batch_size

        for i in range(len(self.activations) - 2, -1, -1):
            batch_size = self.outputs[i].shape[0]
            # print('batch_size : ', batch_size)
            if i < self.num_bn_layers:
                delta = self.activations[i].derivative() * (np.dot( dErr, self.W[i+1].T ))
                # print('delta: ', delta, delta.shape)
                dErr = self.bn_layers[i].backward(delta)
                # print('bckwrd: ', bn_bcw, bn_bcw.shape)
                
                # dErr = bn_bcw * (np.dot( dErr, self.W[i+1].T ))
            else:
                delta = self.activations[i].derivative()
                # print('delta non batch norm: ', delta)
                dErr = self.activations[i].derivative() * (np.dot( dErr, self.W[i+1].T ))
            # print('dErr : ', dErr)
            self.dW[i] = np.dot(self.outputs[i].T, dErr) / batch_size
            self.db[i] = np.dot(np.ones((batch_size, )).T, dErr) / batch_size

    def __call__(self, x):
        return self.forward(x)

    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False


def get_training_stats(mlp, dset, nepochs, batch_size):

    train, val, test = dset
    trainx, trainy = train
    valx, valy = val
    testx, testy = test

    idxs = np.arange(len(trainx))

    training_losses = []
    training_errors = []
    validation_losses = []
    validation_errors = []

    # Setup ...

    for e in range(nepochs):

        # Per epoch setup ...

        for b in range(0, len(trainx), batch_size):

            pass  # Remove this line when you start implementing this
            # Train ...

        for b in range(0, len(valx), batch_size):

            pass  # Remove this line when you start implementing this
            # Val ...

        # Accumulate data...

    # Cleanup ...

    for b in range(0, len(testx), batch_size):

        pass  # Remove this line when you start implementing this
        # Test ...

    # Return results ...

    # return (training_losses, training_errors, validation_losses, validation_errors)

    raise NotImplemented
