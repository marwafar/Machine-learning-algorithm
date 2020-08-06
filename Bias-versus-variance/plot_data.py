#%%
from matplotlib import pyplot as plt
from scipy.io import loadmat
import numpy as np

df=loadmat('ex5data1.mat')
data_set,feature=df['y'].shape
x_0=np.ones(data_set)
x_t=np.insert(df['X'],0,x_0, axis=1)
print(df)

# %%
plt.plot(df['X'],df['y'], marker='o', linestyle='')
plt.xlabel('Change in water level (x)')
plt.ylabel('water flowing out of the dam (y)')
plt.show()

# %%
cost_val=np.genfromtxt('cost_val.csv',delimiter=',')
plt.plot(cost_val[:,0], cost_val[:,1])
plt.show()

# %%
theta=np.array([12.99329412,0.36721182])
y_predict=x_t.dot(theta)
plt.plot(df['X'],df['y'], marker='o', linestyle='')
plt.plot(df['X'],y_predict)
plt.xlabel('Change in water level (x)')
plt.ylabel('water flowing out of the dam (y)')
plt.show()

# %%
learn=np.genfromtxt('learn_curve.csv',delimiter=',')
plt.plot(learn[:,0], learn[:,1])
plt.plot(learn[:,0], learn[:,2])
plt.xlabel('Number of training example')
plt.ylabel('Error')
plt.show()

# %%
cost_val=np.genfromtxt('cost_val_poly.csv',delimiter=',')
plt.plot(cost_val[:,0], cost_val[:,1])
plt.show()

# %%
theta_poly=np.genfromtxt('theta-poly.csv' ,delimiter=',')

degree=9
x_f=np.zeros((data_set,degree))
min_val=np.min(df['X'][:,0])
max_val=np.max(df['X'][:,0])
s=np.linspace(min_val,max_val,data_set)

for i in range(degree):
    x_f[:,i]=np.power(s,i)

mu=np.mean(x_f,axis=0)
x_f[:,1:]=x_f[:,1:]-mu[1:]
sigma=np.std(x_f,axis=0)
x_f[:,1:]=x_f[:,1:]/sigma[1:]

y_predict=x_f.dot(theta_poly)

x_train=df['X'][:,0]
mu=np.mean(x_train)
x_train=x_train-mu
sigma=np.std(x_train)
x_train=x_train/sigma

plt.plot(x_train,df['y'], marker='o', linestyle='')
plt.plot(x_f[:,1],y_predict,linestyle='--')
plt.xlabel('Change in water level (x)')
plt.ylabel('water flowing out of the dam (y)')
plt.show()


# %%
learn=np.genfromtxt('learn_curve_poly.csv',delimiter=',')
plt.plot(learn[:,0], learn[:,1])
plt.plot(learn[:,0], learn[:,2])
plt.xlabel('Number of training example')
plt.ylabel('Error')
plt.show()

# %%
lambda_x=np.genfromtxt("select_lambda.csv", delimiter=',')
plt.plot(lambda_x[:,0], lambda_x[:,1])
plt.plot(lambda_x[:,0],lambda_x[:,2])
plt.xlabel('lambda')
plt.ylabel('Error')
plt.show()

# %%
