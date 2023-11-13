# Titanic-Survival-Prediction-and-Logistic-Regression
This Python code is a data analysis and logistic regression model for predicting the survival of passengers on the Titanic. It demonstrates various steps in the machine learning pipeline, including data analysis, data cleaning, normalization, and logistic regression.
The code is well-documented and provides insights into the following processes:

# Data Analysis:
Loads Titanic dataset from a CSV file.
Displays the first few rows of the dataset.
Checks the dataset's shape and information.
Identifies missing values and data types.
Shows the distribution of passengers by gender and class.

# Data Visualization:
Visualizes the survival rate in different passenger classes.
Visualizes the survival rate by gender.
Generates a scatter matrix to explore relationships between variables.

# Data Cleaning:
Drops unnecessary columns such as 'PassengerId', 'Name', 'Ticket', and 'Cabin'.
Creates dummy variables for categorical columns 'Sex' and 'Embarked'.
Handles missing values in the 'Age' column by filling with the mean.

# Data Normalization:
Normalizes the data to scale all numeric features between 0 and 1.

# Logistic Regression:
Prepares the dataset for logistic regression by defining feature columns and the target variable.
Splits the dataset into training and testing sets.
Creates a logistic regression model and fits it to the training data.
Evaluates the model's accuracy and performance with different thresholds.
Generates a confusion matrix and classification report.

# Cross-Validation:
Performs k-fold cross-validation to evaluate model performance with varying train-test splits.
The code provides a comprehensive example of using logistic regression for binary classification and demonstrates how to evaluate the model's accuracy and robustness using cross-validation. You can use or adapt this code for your own classification tasks.

Feel free to use this code as a starting point for your logistic regression projects and explore different machine learning techniques for predictive modeling.

