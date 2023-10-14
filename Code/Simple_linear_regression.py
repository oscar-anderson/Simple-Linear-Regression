## Simple linear regression.

'''
This basic script uses NumPy and Matplotlib to perform a simple linear regression on
randomly generated data, and visualizes the regression line over a scatter plot.
'''

# Import necessary libraries.
import numpy as np
import matplotlib.pyplot as plt

# Set number of data points to be simulated for both variables.
numValues = 50

# Generate data x-values.
xStart = 0 # Set starting x-value.
xEnd = 10 # Set ending x-value.
x = np.linspace(xStart, xEnd, numValues) # Generate array of values using given parameters.

# Generate random data y-value noise.
noiseMean = np.random.randint(10) # Generate random mean value.
noiseStdDev = np.random.randint(10) # Generate random standard deviation value.
noise = np.random.normal(noiseMean, noiseStdDev, numValues) # Generate array of values using given parameters.

# Generate noisy data y-values.
yStart = 0 # Set starting y-value.
yEnd = np.random.randint(-10, 10) # Use random ending y-value to simulate positive/negative trending data.
y = np.linspace(yStart, yEnd, numValues) + noise # Generate array of values using given parameters

# Initialise array to store linear regression prediction values.
predictions = np.zeros_like(y) # Create array of zeros of same size as y.

# Get parameters of data for regression computation.
xMean = np.mean(x) # Get mean of data x-values.
yMean = np.mean(y) # Get mean of data y-values.

# Perform least squares method.
slope = np.sum((x - xMean) *  (y - yMean)) / np.sum((x - xMean)**2) # Get slope of data.
intercept = yMean - (slope * xMean) # Get y-intercept of data.

# Display regression computation parameters, for transparency.
print('Slope: ', slope) # Display slope.
print('Intercept: ', intercept) # Display y-intercept.

# Perform simple linear regression equation.
predictions = intercept + (slope * x) # Produce regression line prediction of dependent variable.

# Plot data and prediction.
plt.scatter(x, y, color = 'blue', label = 'Data') # Plot data.
plt.plot(x, predictions, color='red', label = 'Prediction') # Plot  regression line.
plt.title('Simple Linear Regression on Randomly Generated Data. \n', fontsize = 11, fontweight = 'bold') # Add title.
plt.xlabel('Indepedent variable') # Add x-axis label.
plt.ylabel('Dependent variable') # Add y-axis label.
plt.xticks(np.arange(xStart, xEnd + 1)) # Include all integers in x-axis ticks.
plt.legend() # Add legend.
plt.grid(True, alpha = 0.5) # Add grid lines.
plt.show() # Display plot.
