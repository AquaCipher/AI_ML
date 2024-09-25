# XGBoost

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
train_dataset = pd.read_csv('train.csv')
test_dataset = pd.read_csv('test.csv')
X_train = train_dataset.iloc[:, :-1].values
y_train = train_dataset.iloc[:, -1].values
X_test = test_dataset.values

# Training XGBoost on the Training set to get feature importances
from xgboost import XGBRegressor
initial_regressor = XGBRegressor(objective='reg:squarederror')
initial_regressor.fit(X_train, y_train)

# Get feature importances and select features
importances = initial_regressor.feature_importances_
threshold = 0.01  # Example threshold
selected_features = np.where(importances > threshold)[0]
X_train_selected = X_train[:, selected_features]
X_test_selected = X_test[:, selected_features]

# Define the grid of hyperparameters to search
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.02, 0.03, 0.04],
    'max_depth': [4, 5, 6],
    'subsample': [0.8, 1],
    'colsample_bytree': [0.8, 1.0],
    'reg_alpha': [0, 0.1, 1, 10],
    'reg_lambda': [0, 0.1, 1, 10]
}

# Set up the grid search with 5-fold cross-validation
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(estimator=initial_regressor, param_grid=param_grid, cv=10, scoring='neg_mean_absolute_error',
                           n_jobs=-1, verbose=2)

# Fit the grid search to the data with selected features
grid_search.fit(X_train_selected, y_train)

# Get the best parameters and MAE score
best_parameters = grid_search.best_params_
best_mae = -grid_search.best_score_
print("Best MAE: {:.2f}".format(best_mae))
print("Best Parameters:", best_parameters)

# Train a new model with the best parameters found using the selected features
best_regressor = XGBRegressor(objective='reg:squarederror', **best_parameters)
best_regressor.fit(X_train_selected, y_train)

# Predicting the Test set results with the best model using the selected features
y_pred = best_regressor.predict(X_test_selected)

# Applying k-Fold Cross Validation for regression with MAE using the selected features
from sklearn.model_selection import cross_val_score
scores = cross_val_score(estimator=best_regressor, X=X_train_selected, y=y_train, cv=10, scoring='neg_mean_absolute_error')
mae_scores = -scores  # Convert to positive MAE scores
print(mae_scores)
print("MAE with Best Parameters and Selected Features: {:.2f} %".format(mae_scores.mean()))
print("Standard Deviation with Best Parameters and Selected Features: {:.2f} %".format(mae_scores.std()))


