import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Seed for reproducibility
np.random.seed(42)

# True linear relationship: y = 2x + 1
def generate_data(x, noise_std=0.5):
    return 2 * x + 1 + np.random.normal(0, noise_std, size=x.shape)

# Case 1: Balanced sampling (5 samples at each concentration)
x_balanced = np.array([0.1, 0.5, 1.0, 2.0, 3.0])
x_balanced = np.repeat(x_balanced, 5)  # 5 samples per level
y_balanced = generate_data(x_balanced)

# Case 2: Biased sampling (20 samples at 1.0, 3 at other levels)
x_biased = np.concatenate([
    np.full(20, 1.0),
    np.full(3, 0.1),
    np.full(3, 0.5),
    np.full(3, 2.0),
    np.full(3, 3.0)
])
y_biased = generate_data(x_biased)

# Fit linear regression models
model_balanced = LinearRegression().fit(x_balanced.reshape(-1, 1), y_balanced)
model_biased = LinearRegression().fit(x_biased.reshape(-1, 1), y_biased)

# Generate test data for predictions
x_test = np.linspace(0, 3.5, 100).reshape(-1, 1)
y_true = 2 * x_test + 1  # true model
y_pred_balanced = model_balanced.predict(x_test)
y_pred_biased = model_biased.predict(x_test)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(x_test, y_true, 'k--', label='True Line (y = 2x + 1)')
plt.plot(x_test, y_pred_balanced, 'b-', label='Balanced Fit')
plt.plot(x_test, y_pred_biased, 'r-', label='Biased Fit')

# Training data points
plt.scatter(x_balanced, y_balanced, color='blue', alpha=0.6, label='Balanced Data')
plt.scatter(x_biased, y_biased, color='red', alpha=0.4, label='Biased Data')

plt.xlabel('Concentration')
plt.ylabel('Sensor Response')
plt.title('Effect of Sampling Bias on Linear Regression Fit')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
