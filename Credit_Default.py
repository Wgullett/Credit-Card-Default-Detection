import pandas as pd
import sklearn as sk
from xgboost import XGBClassifier
from matplotlib import pyplot
from hyperopt import fmin, tpe, hp, STATUS_OK
import numpy as np

# Load the dataset
df = pd.read_csv("VScode/default_credit_cards.csv", index_col=0)
print(df)

# Divide the dataset into features and target variable
df = df.iloc[1:]
Y = df[df.columns.values[23]]
Y = Y.astype(int)
X = df.drop(df.columns.values[23], axis=1)
X = X.astype(int)

# Create validation and test sets
# 80% training, 10% validation, 10% test
total = X.shape[0]
num_test = int(total*0.8)
num_val = int(total*0.1)

X_train = X.iloc[:num_test]
Y_train = Y.iloc[:num_test]

X_test = X.iloc[num_test:num_test+num_val].reset_index()
X_test = X_test.drop(X_test.columns.values[0], axis=1)

Y_test = Y.iloc[num_test:num_test+num_val].reset_index()
Y_test = Y_test.drop(Y_test.columns.values[0], axis=1)

X_val = X.iloc[num_test+num_val:].reset_index()
X_val = X_val.drop(X_val.columns.values[0], axis=1)

Y_val = Y.iloc[num_test+num_val:].reset_index()
Y_val = Y_val.drop(Y_val.columns.values[0], axis=1)

print("Train shape: ", X_train.shape)
print("Test shape: ", X_test.shape)
print("Validation shape: ", X_val.shape)


# Create the model using XGBoost
# XGBoost is a gradient boosting framework that uses tree boosting
model = XGBClassifier(objective='binary:logistic')

# Fit the model to the training data
model.fit(X_train, Y_train)

# Make predictions on the test data
predictions = model.predict(X_test)

# Calculate the accuracy of the model
actual = list(Y_test['Y'])
total = len(actual)
sum = 0
for i in range(total):
    if predictions[i] == actual[i]:
        sum+=1

print("Before Bayesian Hyperparameter Tuning: " + str(sum/total))

#Hyperparameter Tuning
# Hyperopt is a Python library for serial and parallel optimization over awkward search spaces, which may include real-valued, discrete, and conditional dimensions.
# It is based on the Tree of Parzen Estimators (TPE) algorithm, which is a Bayesian optimization algorithm that models the objective function as a Gaussian process.
space = {
    'max_depth' : hp.choice('max_depth', np.arange(1,20, dtype=int)),
    'learning_rate' : hp.loguniform('learning_rate', -5, 3),
    'subsample' : hp.uniform('subsample', 0.5, 1)
}

# The objective function takes a set of hyperparameters as input and returns a loss value to minimize
def objective(params):
    xgb_model = XGBClassifier(**params)
    xgb_model.fit(X_train, Y_train)
    y_pred = xgb_model.predict(X_val)
    score = sk.metrics.accuracy_score(Y_val, y_pred)
    return {'loss' : -score, 'status': STATUS_OK}

# The fmin function is used to minimize the objective function. It takes the objective function, the search space, the optimization algorithm (TPE), and the maximum number of evaluations as input.
best_params = fmin(objective, space, algo=tpe.suggest, max_evals=100)

# Load the best hyperparameters into the model and fit it to the training data
best_model = XGBClassifier(**best_params)
print("BEST HYPER: ", best_params)
best_model.fit(X_train, Y_train)

# Make predictions on the test data using the best model
predictions = best_model.predict(X_test)

# Calculate the accuracy of the best model
actual = list(Y_test['Y'])
total = len(actual)
sum = 0
for i in range(total):
    if predictions[i] == actual[i]:
        sum+=1

print("After Bayesian Hyperparameter Tuning: " + str(sum/total))
