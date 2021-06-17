import numpy as np
import scipy.io
import matplotlib.pyplot as plt




def relu(x):
    temp = np.where(x > 0, x, 0)
    return temp


def sigmoid(x):
    return 1 / (1 + np.exp(x))


def printEights(Z, A1, A2, B1,B2):
    W1 = np.dot(A1, Z[:].T) + B1
    Z1 = relu(W1)
    W2 = np.dot(A2, Z1[:]) + B2
    X = sigmoid(W2)
    return X.T


Z = np.random.normal(0, 1, (100, 10))
mat = scipy.io.loadmat('data21.mat')

X_2D = np.zeros((784, 100))
A1 = mat['A_1']
A2 = mat['A_2']
B1 = mat['B_1']
B2 = mat['B_2']
# print(np.shape(Z))
# print(np.shape(A1), np.shape(B1), np.shape(A2), np.shape(B2))


X = printEights(Z, A1, A2, B1, B2)
for i in range(1, len(Z) + 1):
    X_2D = np.reshape(X[i - 1], (28, 28))
    plt.subplot(10, 10, i)
    plt.axis('off')
    plt.imshow(X_2D.T)

plt.show()