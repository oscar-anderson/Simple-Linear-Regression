"""
Simple Linear Regression example.

This script demonstrates simple linear regression on randomly generated data.
It includes functions for data generation, linear regression computation,
and visualisation using matplotlib and statsmodels.
"""

# Import dependencies
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan

def generate_data(num_values: int = 50, x_start: float = 0, x_end: float = 10) -> tuple:
    """Generate random data with noise.

    Parameters:
    - num_values (int): Number of data points to generate.
    - x_start (float): Starting value for the independent variable.
    - x_end (float): Ending value for the independent variable.

    Returns:
    tuple: Two NumPy arrays representing the independent and dependent variables.
    """
    x = np.linspace(x_start, x_end, num_values)

    # Generate random noise with specified mean and standard deviation.
    noise_mean = np.random.randint(10)
    noise_std_dev = np.random.randint(10)
    noise = np.random.normal(noise_mean, noise_std_dev, num_values)

    # Create dependent variable with random slope and noise.
    y_start = 0
    y_end = np.random.randint(-10, 10)
    y = np.linspace(y_start, y_end, num_values) + noise

    return x, y

def compute_regression_parameters(x: np.ndarray, y: np.ndarray) -> tuple:
    """Compute the slope and intercept for linear regression.

    Parameters:
    - x (np.ndarray): Independent variable data.
    - y (np.ndarray): Dependent variable data.

    Returns:
    tuple: Computed slope and intercept for linear regression.
    """
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    # Compute slope and intercept
    slope = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean)**2)
    intercept = y_mean - (slope * x_mean)

    return slope, intercept

def perform_linear_regression(x: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    """Perform linear regression based on given slope and intercept.

    Parameters:
    - x (np.ndarray): Independent variable data.
    - slope (float): Computed slope for linear regression.
    - intercept (float): Computed intercept for linear regression.

    Returns:
    np.ndarray: Predicted values based on linear regression.
    """
    predictions = intercept + (slope * x)
    return predictions

def display_regression_parameters(slope: float, intercept: float) -> None:
    """Display the computed slope and intercept.

    Parameters:
    - slope (float): Computed slope for linear regression.
    - intercept (float): Computed intercept for linear regression.
    """
    print('Slope:', slope)
    print('Intercept:', intercept)

def plot_data_and_regression(x: np.ndarray, y: np.ndarray, predictions: np.ndarray) -> None:
    """Plot the original data and the regression line.

    Parameters:
    - x (np.ndarray): Independent variable data.
    - y (np.ndarray): Dependent variable data.
    - predictions (np.ndarray): Predicted values based on linear regression.
    """
    plt.scatter(x, y, color = 'blue', label = 'Data')
    plt.plot(x, predictions, color ='red', label = 'Prediction')
    plt.title('Simple Linear Regression on Randomly Generated Data.\n', fontsize = 11, fontweight='bold')
    plt.xlabel('Independent variable')
    plt.ylabel('Dependent variable')
    plt.xticks(np.arange(min(x), max(x) + 1))
    plt.legend()
    plt.grid(True, alpha = 0.5)
    plt.show()
    
def analyse_regression(x: np.ndarray, y: np.ndarray) -> None:
    """Perform linear regression analysis.

    Parameters:
    - x (np.ndarray): Independent variable data.
    - y (np.ndarray): Dependent variable data.
    """
    x_with_intercept = sm.add_constant(x)

    model = sm.OLS(y, x_with_intercept)
    results = model.fit()

    print(results.summary())
    
def plot_residuals(x: np.ndarray, residuals: np.ndarray) -> None:
    """Plot the residuals of the linear regression.

    Parameters:
    - x (np.ndarray): Independent variable data.
    - residuals (np.ndarray): Residuals of the linear regression.
    """
    plt.scatter(x, residuals, color = 'green', label = 'Residuals')
    plt.axhline(y = 0, color = 'black', linestyle = '--', label = 'Zero Residuals Line')
    plt.title('Residuals of Linear Regression\n', fontsize = 11, fontweight = 'bold')
    plt.xlabel('Independent variable')
    plt.ylabel('Residuals')
    plt.xticks(np.arange(min(x), max(x) + 1))
    plt.legend()
    plt.grid(True, alpha = 0.5)
    plt.show()

def simulateRegression() -> None:
    """Main function to run the linear regression example."""
    # Generate data.
    num_values = 1000
    x, y = generate_data(num_values)

    # Fit model to data.
    slope, intercept = compute_regression_parameters(x, y)
    display_regression_parameters(slope, intercept)
    predictions = perform_linear_regression(x, slope, intercept)
    plot_data_and_regression(x, y, predictions)
    
    # Statistically analyse model.
    analyse_regression(x, y)

    # Calculate residuals.
    residuals = y - predictions
    plot_residuals(x, residuals)

# Call main function to run simulation.
simulateRegression()
