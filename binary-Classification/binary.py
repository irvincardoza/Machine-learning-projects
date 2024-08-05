import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import math
file_path='datas.txt'
data = np.loadtxt(file_path, delimiter=',', dtype=float)
# print(data.head())

X = data[:, :-1]  # All rows, all columns except the last
y = data[:, -1] 
print("X_train:")
print(X)
print("y_train:")
print(y)


plt.figure(figsize=(10, 6))

# Plot positive examples
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='blue', marker='o', label='Admitted')

# Plot negative examples
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], c='red', marker='x', label='Not Admitted')

plt.xlabel('Exam 1 Score')
plt.ylabel('Exam 2 Score')
plt.title('Exam Scores vs Admission Decision')
plt.legend()
plt.grid(True)
plt.show()
def sigmoid(z):
 

    g = 1 / (1 + np.exp(-z))
    
    return g

def compute_cost(X, y, w, b, *argv):

    m, n = X.shape

    total_cost = 0.0
    for i in range(m):
        z_i = np.dot(X[i],w) + b
        f_wb_i = sigmoid(z_i)
        total_cost +=  -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i)
             
    total_cost = total_cost / m


    return total_cost

m, n = X.shape

# Compute and display cost with w and b initialized to zeros
initial_w = np.zeros(n)
initial_b = 0.
cost = compute_cost(X, y, initial_w, initial_b)
print('Cost at initial w and b (zeros): {:.3f}'.format(cost))


def compute_gradient(X, y, w, b, *argv): 

    m, n = X.shape
    dj_dw = np.zeros(w.shape)
    dj_db = 0.

    for i in range(m):
        z_wb = np.dot(w, X[i]) + b
        f_wb = sigmoid(z_wb)
        
        dj_db_i = f_wb -y[i]
        dj_db += dj_db_i
        
        for j in range(n):
            dj_dw_ij = (f_wb -y[i]) * X[i,j]
            dj_dw[j] += dj_dw_ij
            
    dj_dw = dj_dw/m
    dj_db = dj_db/m


        
    return dj_db, dj_dw

initial_w = np.zeros(n)
initial_b = 0.

dj_db, dj_dw = compute_gradient(X, y, initial_w, initial_b)
print(f'dj_db at initial w and b (zeros):{dj_db}' )
print(f'dj_dw at initial w and b (zeros):{dj_dw.tolist()}' )

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters, lambda_): 

    # number of training examples
    m = len(X)
    
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w_history = []
    
    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db, dj_dw = gradient_function(X, y, w_in, b_in, lambda_)   

        # Update Parameters using w, b, alpha and gradient
        w_in = w_in - alpha * dj_dw               
        b_in = b_in - alpha * dj_db              
       
        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            cost =  cost_function(X, y, w_in, b_in, lambda_)
            J_history.append(cost)

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters/10) == 0 or i == (num_iters-1):
            w_history.append(w_in)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}   ")
        
    return w_in, b_in, J_history, w_history #return w and J,w history for graphing

np.random.seed(1)
initial_w = 0.01 * (np.random.rand(2) - 0.5)
initial_b = -8

# Some gradient descent settings
iterations = 10000
alpha = 0.001

w,b, J_history,_ = gradient_descent(X ,y, initial_w, initial_b, 
                                   compute_cost, compute_gradient, alpha, iterations, 0)

# UNQ_C4
# GRADED FUNCTION: predict

def predict(X, w, b): 

    m, n = X.shape   
    p = np.zeros(m)

    for i in range(m):   
        z_wb = 0
        # Loop over each feature
        for j in range(n): 
            # Add the corresponding term to z_wb
           
            z_wb_ij = X[i, j] * w[j]
            z_wb += z_wb_ij
        
        # Add bias term 
        z_wb += b
        
        # Calculate the prediction for this example
        f_wb = f_wb = sigmoid(z_wb)

        # Apply the threshold
        p[i] = f_wb >= 0.5
        
    ### END CODE HERE ### 
    return p

p = predict(X, w,b)
print('Train Accuracy: %f'%(np.mean(p == y) * 100))
