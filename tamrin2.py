import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib

df = pd.read_csv("./data.csv")
df = shuffle(df)
df = df.values
data = df[:,:2]
labels = df[:,2]


#-----------Scatter Data---------------------#

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=42)
colors = ['red','blue']

print(X_train.shape, y_train.shape)
plt.figure(figsize = (6,16))
plt.subplot(2, 1, 1)
plt.scatter(X_train[:,0], X_train[:,1], c=y_train, cmap=matplotlib.colors.ListedColormap(colors))
plt.legend()
plt.title('train data')
plt.subplot(2, 1, 2)
plt.scatter(X_test[:,0], X_test[:,1], c=y_test, cmap=matplotlib.colors.ListedColormap(colors))
plt.title('test data')
plt.show()

#---------------Imelement neural network by a hidden layer--------------------------------#


#print(X.shape,y.shape,data.shape,labels.shape)
num_i_units = 2
num_h_units = 2
num_o_units = 1

learning_rate = 0.01# 0.001, 0.01 <- Magic values
n_epochs = 12000 # 5000 <- Magic value
m = X_train.shape[0]# Number of training examples

# The model needs to be over fit to make predictions. Which 
np.random.seed(1)
W = np.random.normal(0, 1, (1, num_i_units)) # 1x2
V = np.random.normal(0, 1, (1, num_i_units)) # 1x2
U = np.random.normal(0, 1, (num_o_units, num_h_units)) # 1x2

B1 = np.array([[1]]) # 1x1
B2 = np.array([[1]]) # 1x1
B3 = np.array([[1]]) # 1x1

def sigmoid(z, derv=False):
    if derv: return z * (1 - z)
    return 1 / (1 + np.exp(-z))

def forward(x, predict=False):
    X = x.reshape(x.shape[0], 1) # Getting the training example as a column vector.

    z1 = W.dot(X) + B1
    z2 = V.dot(X) + B2
    Z[0][0] = sigmoid(z1)
    Z[1][0] = sigmoid(z2)
    

    UZ = U.dot(Z) + B3 # 1x2 * 2x1 + 1x1 = 1x1
    y_pred = sigmoid(UZ) # 1x1

    return y_pred

dW = 0
dU = 0

dB1 = 0
dB2 = 0
dB3 = 0 

cost = np.zeros((n_epochs, 1))
for epoch in range(n_epochs):
    c = 0
    dW = 0
    dV = 0
    dU = 0

    dB1 = 0
    dB2 = 0
    dB3 = 0
    Z = np.zeros((2,1))
    
    for j in range(m):
        

        # Forward Prop.
        X = X_train[j].reshape(X_train[j].shape[0], 1) # 2x1

        z1 = W.dot(X) + B1
        z2 = V.dot(X) + B2
        Z[0][0] = sigmoid(z1)
        Z[1][0] = sigmoid(z2)
        

        UZ = U.dot(Z) + B3
        y_pred = sigmoid(UZ) 

        # Back prop.
        dUZ = y_pred - labels[j] # 1x1
        dU += dUZ *sigmoid(y_pred,derv=True)* Z.T 

        dWX = U[0][0] * dUZ * sigmoid(y_pred,derv=True) *sigmoid(Z[0][0], derv=True)  
        dvx = U[0][1] * dUZ * sigmoid(y_pred,derv=True) *sigmoid(Z[1][0], derv=True) 
       
        
        k = np.array([[dWX[0][0]],[dvx[0][0]]])
   
        dW += dWX * X.T
        dV += dvx * X.T

        

        dB1 += dWX
        dB2 +=  dvx
        dB3 += dUZ 
       
        c = c + (0.5 * (y_pred-y_train[j])*(y_pred-y_train[j]))
        sys.stdout.flush() # Updating the text.

    
        
    W= W - learning_rate * (dW / m) 
    V= V - learning_rate * (dV / m) 
    U = U - learning_rate * (dU / m)

    B1 = B1 - learning_rate * (dB1 / m)
    B2 = B2 - learning_rate * (dB2 / m)
    B3 = B3 - learning_rate * (dB3 / m)
    cost[epoch] = (c / m) 

    if epoch % 500 ==0:
    	print("epoch# ",epoch," cost: ",cost[epoch])

#---------------Calculate accuracy---------------------#
cor=0
pred_labels = np.zeros(y_test.shape[0])
for id in range(X_test.shape[0]):
    
    x = X_test[id]
    y_pred = forward(x, predict=True)
    y_pred = np.round(y_pred)
    pred_labels[id] = y_pred
    if y_pred==y_test[id]:
    	cor+=1

print('test accuracy',float(cor)/X_test.shape[0] )


#-----------------------------scatter output of network ----------------------------#
plt.figure(figsize = (6,16))
plt.subplot(2, 1, 1)
plt.scatter(X_test[:,0], X_test[:,1], c=y_test, cmap=matplotlib.colors.ListedColormap(colors))
plt.title('true labels')
plt.subplot(2, 1, 2)
plt.scatter(X_test[:,0], X_test[:,1], c=pred_labels, cmap=matplotlib.colors.ListedColormap(colors))
plt.title('predict labels')
plt.show()


plt.plot(range(n_epochs), cost)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.show()
