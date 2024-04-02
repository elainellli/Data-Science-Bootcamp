import numpy as np
import matplotlib.pyplot as plt

x1 = [0.49, 1.69, 0.04, 1.0, 0.16, 0.25, 0.49, 0.04]
x2 = [0.09, 0.04, 0.64, 0.16, 0.09, 0.0, 0.0, 0.01]
# h_w = [0.389, 0.042, 0.613, 0.167, 0.572, 0.526, 0.393, 0.638]
# y_true = [0, 0, 0, 0, 1, 1, 1, 1]
#



x = np.array([[1, 0.49, 0.09], [1, 1.69, 0.04], [1, 0.04, 0.64], [1, 1.0, 0.16], [1, 0.16, 0.09], [1, 0.25, 0.0], [1, 0.49, 0.0], [1, 0.04, 0.01]])
y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
# Given weight vector
w_prime = np.array([0.668, -2.239, -0.185])

# Learning rate
alpha = 0.1

# Calculate predicted probabilities using w_prime
predicted_probs = 1 / (1 + np.exp(-np.dot(x, w_prime)))

# Calculate gradient of the logistic loss function
gradient = np.mean((y_true - predicted_probs)[:, np.newaxis] * x, axis=0)

# Update weights using gradient ascent
w_prime_new = w_prime + alpha * gradient

# print("Updated weight vector after one step of gradient ascent:")
# print(w_prime_new)
h_w_new = []
L_w = 1
for i in range(len(x1)):
    sigma_wtx = 1 / (1 + np.exp(-(0.668 - 2.239* x1[i] - 0.185 * x2[i])))  # Sigmoid function
    h_w_new.append(sigma_wtx)
    L_w *= (sigma_wtx ** y_true[i]) * ((1 - sigma_wtx) ** (1 - y_true[i]))

print(h_w_new)

cross_entropy_loss = 0
for i in range(len(x1)):
    y_pred = h_w_new[i]  # Predicted probability of the positive class
    cross_entropy_loss += y_true[i] * np.log(y_pred) + (1 - y_true[i]) * np.log(1 - y_pred)

print(cross_entropy_loss)
# # Print the result
# print("Logistic Loss (L(w)): {:.8f}".format(L_w))

