# Simple Linear Regression example

This repository contains a Python script demonstrating simple linear regression on randomly generated data. The script utilises functions for data generation, linear regression computation, and visualisation using Matplotlib and Statsmodels. The purpose of this script was to allow me to develop and practice my skills in producing clean Python code, as well as test my mathematical understanding of linear regression.

## Overview

Simple linear regression is a statistical method that models the relationship between two variables by fitting a linear equation to observed data. In this example, we generate random data with noise and perform linear regression to predict the dependent variable based on the independent variable.

## Features

- **Data Generation:** The `generate_data` function creates random data with specified parameters, including the number of data points and the range of the independent variable.

- **Linear Regression Computation:** The `compute_regression_parameters` function calculates the slope and intercept for linear regression based on the provided data.

- **Prediction:** The `perform_linear_regression` function uses the computed slope and intercept to make predictions.

- **Visualisation:** The script includes functions for plotting the original data and the regression line (`plot_data_and_regression`), as well as visualising residuals (`plot_residuals`).

- **Statistical Analysis:** The `analyse_regression` function utilises Statsmodels to perform a detailed statistical analysis of the linear regression model.

## Usage

1. Ensure you have Python and the required libraries (NumPy, Matplotlib, Statsmodels) installed:

   ```
   pip install numpy matplotlib statsmodels
   ```

2. Clone the repository:

  ```
  git clone https://github.com/oscar-anderson/simple-linear-regression.git
  ```

3. Run the script:

  ```
  cd simple-linear-regression
  python linear_regression_example.py
  ```

## Results
The script generates a plot showing the original data points in blue and the regression line in red. Additionally, statistical analysis results and a plot of residuals are displayed to provide insights into the model's performance.

## Acknowledgements

- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [Statsmodels](https://www.statsmodels.org/)
