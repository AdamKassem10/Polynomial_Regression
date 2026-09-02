# Polynomial Regression

Predicting salary based on position level using a polynomial (curved) regression model.

## Overview

This project compares a simple linear regression model against a polynomial regression model to see which one fits the salary data better. Since salary jumps sharply at higher position levels, a straight line underfits the data, a curved line fits it much more closely.

The equation the polynomial model is based on:

y = b0 + b1x1 + b2x1^2 + ... + bn*x1^n


It's still called linear regression because it's linear with respect to the coefficients (b0, b1, b2, etc), not the x variable. The x variable gets squared, cubed, and so on, but the coefficients themselves are still combined in a linear way.

## Dataset

`Position_Salaries.csv` contains 10 rows with the following columns:
- `Position`
- `Level`
- `Salary`

## What the script does

1. Imports the dataset, using only the Level column as the feature
2. Trains a Linear Regression model on the whole dataset for comparison
3. Trains a Polynomial Regression model (degree 4) on the whole dataset
4. Visualizes the linear regression fit
5. Visualizes the polynomial regression fit, including a higher resolution smoother curve
6. Predicts a salary for position level 6.5 with both models

## Results

Predicting the salary for level 6.5:
- Linear Regression predicts **$330,378.79**
- Polynomial Regression predicts **$158,862.45**

The polynomial prediction is much closer to the actual trend in the data, showing why a curved model fits this dataset better than a straight line.

## Tech used

- Python
- NumPy
- Pandas
- Matplotlib
- scikit-learn

## How to run

pip install numpy pandas matplotlib scikit-learn
python polynomial_regression.py
