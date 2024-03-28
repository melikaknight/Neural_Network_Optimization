import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
import pandas as  pd
from sklearn.model_selection import train_test_split


#Tamrin 1
#Load data
df = pd.read_csv("./data.csv")
df = shuffle(df)
data = df.values
X = data[:,:2]
y = data[:,2]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

colors = ['red','blue']


#-----------Scatter Data---------------------#
plt.figure(figsize = (6,16))
plt.subplot(2, 1, 1)
plt.scatter(X_train[:,0], X_train[:,1], c=y_train, cmap=matplotlib.colors.ListedColormap(colors))
plt.legend()
plt.title('train data')
plt.subplot(2, 1, 2)
plt.scatter(X_test[:,0], X_test[:,1], c=y_test, cmap=matplotlib.colors.ListedColormap(colors))
plt.title('test data')
plt.show()


#---------------Imelement neural--------------------------------#
#Tamrin 3
def sigmoid(x):
    return 1/(1+np.exp(-x))

#calculates the derivative of the sigmoid function
#The derivative of sigmoid function is simply sigmoid(x) * sigmoid(1-x)
def sigmoid_der(x):
    return sigmoid(x)*(1-sigmoid(x))


#dimation of X_train dimation
input_dim = 2
#initialize weights and bias
np.random.seed(42)
W = np.random.rand(input_dim,1)
B = np.random.rand(1,1)

n_epochs = 5000
lr = 0.01


X_train = X_train
y_train = y_train


cost = np.zeros((n_epochs, 1))

m = X_train.shape[0]
for epoch in range(n_epochs):

	w_grad = 0
	b_grad = 0
	c=0
	
	for id,X in enumerate(X_train):

		#compute y
		XW = np.dot(X, W) + B
		y_pred = sigmoid(XW)

		#compute cost
		#print(y_pred, y_train[id])
		error = y_pred - y_train[id]
		
		#need to differentiate from cost function(MSE) respect to weights
		#compute d_cost/d_w = d_cost/d_pred * d_pred/d_z * d_z/d_w
		dcost_dpred = error
		dpred_dz = sigmoid_der(XW)
		#d_z/d_w -> z = w1x1 + w2x2 + b: (x1,x2) = X
		z_delta = dcost_dpred * dpred_dz
		z_delta = z_delta.reshape(z_delta.shape[0],1)
		
		X = X.T.reshape(X.shape[0],1)
		dcost_dw = np.dot(X, z_delta)
		
		w_grad += dcost_dw
		b_grad += z_delta

		c = c + (0.5 * (error)*(error))


	#update weights and bais
	W -= lr * w_grad*(1./m)
	B -= lr* b_grad*(1./m)

	cost[epoch] = (c / m)
	if epoch % 500 ==0:
		print("epoch# ",epoch," cost: ",cost[epoch])


#Tamrin 4
#---------------Calculate accuracy---------------------#
cor=0
pred_labels = np.zeros(y_test.shape[0])
for id in range(X_test.shape[0]):
    
    X = X_test[id]
    XW = np.dot(X, W) + B
    y_pred = sigmoid(XW)
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

