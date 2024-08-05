
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import math
import copy


data = pd.read_csv('Student_Performance.csv', names = ['Hours Studied', 'Previous Scores','Extracurricular Activities','Sleep Hours','Sample Question Papers Practiced','Performance Index'])

data['Hours Studied'] = pd.to_numeric(data['Hours Studied'], errors='coerce')
data['Previous Scores'] = pd.to_numeric(data['Previous Scores'], errors='coerce')

data['Sleep Hours'] = pd.to_numeric(data['Sleep Hours'], errors='coerce')

data['Extracurricular Activities'] = data['Extracurricular Activities'].replace({'No': 0, 'Yes': 1})

data['Sample Question Papers Practiced'] = pd.to_numeric(data['Sample Question Papers Practiced'], errors='coerce')
data['Performance Index'] = pd.to_numeric(data['Performance Index'], errors='coerce')

data = data.dropna()

data['Extracurricular Activities'] = data['Extracurricular Activities'].astype(int)


X = data[['Hours Studied', 'Previous Scores', 'Extracurricular Activities', 'Sleep Hours', 'Sample Question Papers Practiced']].values
y = data['Performance Index'].values



print('Features (X):', X.shape)
print('Target (y):', y.shape)
# print(data.head())
num_features=5
w_init = np.zeros(num_features)  # or use np.random.randn(num_features) for random initialization
b_init = 0  # or use a small random value if desired

# we can a dot product function
def predict(x, w, b): 
  
    p = np.dot(x, w) + b     
    return p   

# function to calculate cost
def compute_cost(X, y, w, b): 
  
    
    m = X.shape[0]
    cost = 0.0
    for i in range(m):                                
        f_wb_i = np.dot(X[i], w) + b           #(n,)(n,) = scalar (see np.dot)
        cost = cost + (f_wb_i - y[i])**2       #scalar
    cost = cost / (2 * m)                      #scalar    
    return cost

def compute_gradient(X, y, w, b): 

   
    m,n = X.shape           #(number of examples, number of features)
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):                             
        err = (np.dot(X[i], w) + b) - y[i]   
        for j in range(n):                         
            dj_dw[j] = dj_dw[j] + err * X[i, j]    
        dj_db = dj_db + err                        
    dj_dw = dj_dw / m                                
    dj_db = dj_db / m                                
        
    return dj_db, dj_dw

# lets analize gradient

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 
    
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in
    
    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db,dj_dw = gradient_function(X, y, w, b)   ##None

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw               ##None
        b = b - alpha * dj_db               ##None
      
        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            J_history.append( cost_function(X, y, w, b))

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")
        
    return w, b, J_history #return final w,b and J history for graphing

cost = compute_cost(X, y, w_init, b_init)
print(f'Cost at optimal w : {cost}')

tmp_dj_db, tmp_dj_dw = compute_gradient(X, y, w_init, b_init)
print(f'dj_db at initial w,b: {tmp_dj_db}')
print(f'dj_dw at initial w,b: \n {tmp_dj_dw}')

initial_w = np.zeros_like(w_init)
initial_b = 0.
# some gradient descent settings
iterations = 1000
alpha = 5.0e-7
# run gradient descent 
w_final, b_final, J_hist = gradient_descent(X, y, initial_w, initial_b,
                                                    compute_cost, compute_gradient, 
                                                    alpha, iterations)
print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
m,_ = X.shape
# for i in range(m):
#     print(f"prediction: {np.dot(X[i], w_final) + b_final:0.2f}, target value: {y[i]}")


# LEARNING CURVE
plt.plot(range(len(J_hist)), J_hist, 'b-', label='Training Cost')
plt.xlabel('Number of Iterations')
plt.ylabel('Cost')
plt.title('Learning Curve')
plt.legend()
plt.grid(True)
plt.show()



# Define new student data to test prediction
new_student = np.array([5, 52, 1, 5, 2])  # feature values
predicted_performance = predict(new_student, w_final, b_final)
print(predicted_performance)



from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Predictions
y_pred = np.dot(X, w_final) + b_final

# Metrics
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.2f}")




















#########################  BELOW IS CODE TO SHOW 3D GRAPH OF GRADIENT DESCENT    ###################################






# from mpl_toolkits.mplot3d import Axes3D

# def plot_cost_surface(X, y, w_final, b_final, J_hist):
#     # Define a grid of w1 and w2 values
#     w1_range = np.linspace(w_final[0] - 1, w_final[0] + 1, 100)
#     w2_range = np.linspace(w_final[1] - 1, w_final[1] + 1, 100)
#     W1, W2 = np.meshgrid(w1_range, w2_range)
#     J = np.zeros_like(W1)

#     # Calculate cost for each (w1, w2) pair
#     for i in range(W1.shape[0]):
#         for j in range(W1.shape[1]):
#             w_temp = np.array([W1[i, j], W2[i, j], w_final[2], w_final[3], w_final[4]])
#             J[i, j] = compute_cost(X, y, w_temp, b_final)

#     # Plotting the cost surface
#     fig = plt.figure(figsize=(12, 8))
#     ax = fig.add_subplot(111, projection='3d')
#     ax.plot_surface(W1, W2, J, cmap='viridis', alpha=0.6)
#     ax.plot_wireframe(W1, W2, J, color='k', linewidth=0.5)
#     ax.scatter(w_final[0], w_final[1], compute_cost(X, y, w_final, b_final), color='r', s=100, label='Final w')
#     ax.set_xlabel('w1')
#     ax.set_ylabel('w2')
#     ax.set_zlabel('Cost')
#     ax.set_title('3D Surface Plot of Cost Function')
#     plt.legend()
#     plt.show()

# def plot_gradient_descent_path(X, y, w_init, b_init, cost_function, gradient_function, alpha, num_iters):
#     w, b, J_history = gradient_descent(X, y, w_init, b_init, cost_function, gradient_function, alpha, num_iters)

#     # Create a grid of w1 and w2 values
#     w1_range = np.linspace(w_init[0] - 1, w_init[0] + 1, 100)
#     w2_range = np.linspace(w_init[1] - 1, w_init[1] + 1, 100)
#     W1, W2 = np.meshgrid(w1_range, w2_range)
#     J = np.zeros_like(W1)

#     # Calculate cost for each (w1, w2) pair
#     for i in range(W1.shape[0]):
#         for j in range(W1.shape[1]):
#             w_temp = np.array([W1[i, j], W2[i, j], w_init[2], w_init[3], w_init[4]])
#             J[i, j] = compute_cost(X, y, w_temp, b_init)

#     # Plotting the cost surface
#     fig = plt.figure(figsize=(12, 8))
#     ax = fig.add_subplot(111, projection='3d')
#     ax.plot_surface(W1, W2, J, cmap='viridis', alpha=0.6)
#     ax.plot_wireframe(W1, W2, J, color='k', linewidth=0.5)

#     # Plot the path of gradient descent
#     w_history = np.array([w_init] + [J_history[i] for i in range(num_iters)])
#     ax.plot(w_history[:, 0], w_history[:, 1], [compute_cost(X, y, w, b_init) for w in w_history], color='r', marker='o', markersize=5, linestyle='-', label='Gradient Descent Path')

#     ax.set_xlabel('w1')
#     ax.set_ylabel('w2')
#     ax.set_zlabel('Cost')
#     ax.set_title('3D Gradient Descent Path')
#     plt.legend()
#     plt.show()

# # Plot cost surface
# plot_cost_surface(X, y, w_final, b_final, J_hist)

# # Plot gradient descent path
# plot_gradient_descent_path(X, y, w_init, b_init, compute_cost, compute_gradient, alpha, iterations)
