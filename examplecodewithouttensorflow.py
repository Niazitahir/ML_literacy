import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
filename = os.path.abspath("data\diagnosis.csv")
data = pd.read_csv(filename, names= ["Temp", "Nausea", "Lumbar", "Urine", "micturition", "burning", "decision1", "decision2"])
data = np.array(data)

m, n = data.shape
np.random.shuffle(data)

#Fucked up, y_dev should be the labels, x_dev should be everything else


data_dev = data[0:int(m/2)].T
Y_dev = data_dev[(n-2):n]
X_dev = data_dev[0:(n-2)]
for i in range(len(Y_dev)):
    x = 0
    for y in Y_dev[i]:
        
        if "no" in y:
            y = 0
        else:
            y = 1
        Y_dev[i][x] = y
        x+=1 
Y_labels = []
for i in range(len(Y_dev[0])):
    temp = [Y_dev[0][i], Y_dev[1][i]]
    if temp == [0,0]:
        temp = 0
    if temp == [0,1]:
        temp = 1
    if temp == [1,0]:
        temp = 2
    if temp == [1,1]:
        temp = 3
    Y_labels.append(temp)
Y_dev = Y_labels





for i in range(len(X_dev)-1):

    for y in range(len(X_dev[i])):
        if "no" in X_dev[i+1][y]:
            temp = 0
        else:
            temp = 1
        X_dev[i+1][y] = temp


i =0
for x in X_dev[0]:
    x = x.replace("," , ".")  
    X_dev[0][i] = float((x))/42
    i+=1


    
    


data_train = data[int(m/2):m].T
Y_train = data_train[(n-2):n]
X_train = data_train[0:(n-2)]
i = 0
for i in range(len(Y_train)):
    x = 0
    for y in Y_train[i]:
        
        if "no" in y:
            y = 0
        else:
            y = 1
        Y_train[i][x] = y
        x+=1 
Y_labels = []
for i in range(len(Y_train[0])):
    temp = [Y_train[0][i], Y_train[1][i]]
    if temp == [0,0]:
        temp = 0
    if temp == [0,1]:
        temp = 1
    if temp == [1,0]:
        temp = 2
    if temp == [1,1]:
        temp = 3
    Y_labels.append(temp)
Y_train = Y_labels

for i in range(len(X_train)-1):
    
    for y in range(len(X_train[i])):
        if "no" in X_train[i+1][y]:
            temp = 0
        else:
            temp = 1
        X_train[i+1][y] = temp
i = 0
for x in X_train[0]:
    try:    
        x = x.replace("," , ".")  
    except:
        x = x
    X_train[0][i] = float((x))/42
    i+=1




#/////////////////////////////////////////////////////////////////////////////////////////

def init_param():
    W1 = np.random.rand(10, 6) * np.sqrt(1/6) #should be shape of x_train, dim(1) W = dim(0) x_train 
    W2 = np.random.rand(4, 10) * np.sqrt(1/10)
    b1 = np.zeros(())
    b2 = np.zeros(())
    
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(Z,0)

def softmax(Z):
    Z = np.float32(Z)
    A = np.exp(Z)/sum(np.exp(Z))
    return A

def forward_prop(W1, b1, W2, b2, X):

    Z1 = W1.dot(X) + b1
    
    A1 = ReLU(Z1)

    Z2 = W2.dot(A1) + b2

    
    A2 = softmax(Z2)

    return Z1, A1, Z2, A2

def ReLU_deriv(Z):
    return Z>0
def one_hot(Y):
    length = len(Y)
    one_hot_Y = np.zeros((length, max(Y) + 1))
    one_hot_Y[np.arange(length), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2
def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2


def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / len(Y)

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_param()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2

W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.1, 150)

    
