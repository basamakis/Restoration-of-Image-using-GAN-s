import numpy as np
import scipy.io
import matplotlib.pyplot as plt


def relu(x):
    temp = np.where(x > 0, x, 0)
    return temp


def sigmoid(x):
    return 1 / (1 + np.exp(x))


def relu_derivative(x):
    temp = np.where(x > 0, 1, 0)
    temp = np.reshape(temp, (len(temp), 1))
    return temp


def sigmoid_derivative(x):
    return -np.exp(x) / np.power((1 + np.exp(x)), 2)


def printEights(Z, A1, A2, B1, B2):
    W1 = np.dot(A1, Z.T) + B1
    Z1 = relu(W1)
    W2 = np.dot(A2, Z1) + B2
    
    X = sigmoid(W2)

    forwardParams = {'X': X.T, 'W2': W2, 'W1': W1}
    return forwardParams


def costFunction(params, T, Z, Xn):
    X = params['X']
    temp1 = np.dot(X, T.T) - Xn
    temp2 = np.sum(np.power(temp1, 2))
    return len(Xn.T) * np.log(temp2) + np.sum(np.power(Z, 2))


def gradientCalc(A1, A2, params, T, Z, Xn):
    X = params['X']
    X = np.dot(X, T.T)
    W2 = params['W2']
    W2 = W2.T
    W2 = np.dot(W2, T.T)
    A2temp = A2.T
    A2temp = np.dot(A2temp, T.T)
    W1 = params['W1']


    # CALCULATION OF U2 and V2
    temp1 = X - Xn
    temp2 = np.sum(np.power(temp1, 2))
    u2 = 2 * (X - Xn) / temp2
    v2 = u2 * sigmoid_derivative(W2)
    # CALCULATION OF U1 and V1
    u1 = np.dot(A2temp, v2.T)
    v1 = u1 * relu_derivative(W1)

    #return the derivative
    
    u0 = np.dot(A1.T, v1)
    return N * u0 + 2 * Z.T


N = 500
selectImg = 0;
T = np.zeros((N, 784))
np.fill_diagonal(T, 1)

mat = scipy.io.loadmat('data22.mat')

Xi = mat['X_i']
Xn = mat['X_n']

mat = scipy.io.loadmat('data21.mat')
A1 = mat['A_1']
A2 = mat['A_2']
B1 = mat['B_1']
B2 = mat['B_2']

Xi = Xi.T
Xn = Xn.T

Z = np.random.normal(0, 1, (1, 10))
Xn0 = np.dot(T, Xn[selectImg])
Xn0 = np.reshape(Xn0, (1, len(Xn0)))




# PRINT IDEAL EIGHT
X_2D = np.reshape(Xi[selectImg], (28, 28))
plt.subplot(1, 3, 1)
plt.axis('off')
plt.imshow(X_2D.T)


# PRINT TRANSFORMED EIGHT
shape = np.shape(Xn0)
padded_array = np.zeros((1, 784))
padded_array[:shape[0],:shape[1]] = Xn0

X_2D = np.reshape(padded_array, (28, 28))
plt.subplot(1, 3, 2)
plt.axis('off')
plt.imshow(X_2D.T)

learning_rate = 0.001
iterations = 2000
COST =[]

for i in range(iterations):
    params = printEights(Z, A1, A2, B1, B2)
    x = params['X']
    COST.append(costFunction(params, T, Z, Xn0))
    grad = gradientCalc(A1, A2, params, T, Z, Xn0)
    grad = grad.T
    Z = Z - learning_rate * grad



# PRINT IDEAL EIGHT
params = printEights(Z, A1, A2, B1, B2)
X_2D = np.reshape(params['X'], (28, 28))
plt.subplot(1, 3, 3)
plt.axis('off')
plt.imshow(X_2D.T)

plt.show()

x = np.linspace(1, len(COST), len(COST))
plt.plot(x, COST)
plt.show()
