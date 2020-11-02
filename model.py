import numpy as np
from mnist import MNIST
import torch
import torch.nn as nn


# Main function:
#     Load data
#         output: X_train, Y_train, X_test, Y_test
#     Initialize Variables
#         output: W[1], W[2], W[3], b[1], b[2], b[3]
#     For Loop:
#         Forward Propagation
#             input: X_train, Y_train, W[1], W[2], W[3], b[1], b[2], b[3]
#             Layer 1: W[1].T * X_train + b[1] -> ReLu -> A[1]
#             Layer 2: W[2].T * A[1] + b[2] -> ReLu -> A[2]
#             Layer 3: W[3].T * A[2] + b[3] -> Softmax -> Y_pred
#             output: W[1], W[2], W[3], b[1], b[2], b[3], Y_pred
#         Loss Calculator
#             input: Y_pred, Y_test
#             Binary cross entropy loss function
#             H(q) = (-1/N) * sum(y*log(y_hat) + (1-y)*log(1-y_hat))
#             output: H(q)
#         Backward Propagation
#             input: W[1], W[2], W[3], b[1], b[2], b[3], Y_pred, H(q)
#             dz = a - y
#             output: W[1], W[2], W[3], b[1], b[2], b[3]

def append(value):
    for x in 60000:
        Y = np.identity(10)[value[x]]
        Y_matrix[value][x] = 1
        print(Y_matrix[value])
    return Y_matrix


def LoadData() :
    mndata = MNIST('samples')
    X_train, Y_train = mndata.load_training()
    X_test, Y_test = mndata.load_testing()
    print(np.identity(10)[9])
    print(Y_train[2])
    #print(append(Y_train))
    return np.transpose(X_train), np.transpose(Y_train), np.transpose(X_test), np.transpose(Y_test)
#X_train, Y_train, X_test, Y_test = LoadData()
#params = initialize(784, 25, 12, 10)
def initialize(inSize, hiddenLayer1, hiddenLayer2, outSize):
    # inSize = 400
    # hiddenLayer1 = 25
    # hiddenLayer2 = 12
    # outSize = 10
    W1 = np.random.randn(hiddenLayer1, inSize)
    B1 = np.zeros((hiddenLayer1, 1))
    W2 = np.random.randn(hiddenLayer2, hiddenLayer1)
    B2 = np.zeros((hiddenLayer2, 1))
    W3 = np.random.randn(outSize, hiddenLayer2)
    B3 = np.zeros((outSize, 1))

    params = {
        "W1" : W1,
        "W2" : W2,
        "W3" : W3,
        "B1" : B1,
        "B2" : B2,
        "B3" : B3
    }
    return params
#initialize(400, 25, 12, 10)

def sigmoid(x):
    sigmoid = 1/(1+np.exp(x))
    return sigmoid

def softmax(x):
    softmax = np.exp(x)/sum(np.exp(x))
    return softmax

def fwdProp(X_train, params):
    Z1 = np.matmul(params["W1"], X_train) + params["B1"]
    A1 = sigmoid(Z1)
    Z2 = np.matmul(params["W2"], A1) + params["B2"]
    A2 = sigmoid(Z2)
    Z3 = np.matmul(params["W3"], A2) + params["B3"]
    Y_hat = softmax(Z3)
    #print("yhat")
    #print(np.shape(Y_hat))
    return Y_hat

def cost(Y_hat, Y_train):
    cost = (-1/len(Y_hat[0])) * sum(y*log(y_hat) + (1-y)*log(1-y_hat))
    return cost

def bwdProp(dZ, cache):
    #how do you cache?!
    dW = 1/m*np.dot(dZ, A_prev.T)
    db = 1/m*(np.sum(dZ, axis=1, keepdims=True))
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db

def chgParams(params, grads, learning_rate):
    #basically just implameters["W1"] = params["ement the following
    #parW1"] - learning_rate*grads[dW1]

#main model:
X_train, Y_train, X_test, Y_test = LoadData()
params = initialize(784, 25, 12, 10)
Y_hat = fwdProp(X_train, params)
cost(Y_hat, Y_train)
#bwdProp(dZ, cache)