# %%
from sklearn.datasets import fetch_california_housing
data = fetch_california_housing(as_frame=True)
df = data.frame


# %%
print(df.head())
print(df.describe())
print(df.info())

# %%
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

# %%
print(df.isnull().sum())

# %%
# Define features and target
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# %%
X_train

# %%
y_train

# %%
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
import pickle

X_reg = df.drop("MedHouseVal", axis=1)
y_reg = df["MedHouseVal"]

# Train-test split for regression
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# Train a baseline RandomForestRegressor
baseline_rf = RandomForestRegressor(random_state=42)
baseline_rf.fit(X_train_reg, y_train_reg)

# Evaluate the regression model
y_pred_reg = baseline_rf.predict(X_test_reg)
mse_reg = mean_squared_error(y_test_reg, y_pred_reg)
print("Baseline Regression MSE:", mse_reg)

# Save the regression model
with open("baseline_rf_model.pkl", "wb") as f:
    pickle.dump(baseline_rf, f)

# %%
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
import pickle



# ----------------------------
# Regression Pipeline
# ----------------------------
# Use the continuous target for regression
X_reg = df.drop("MedHouseVal", axis=1)
y_reg = df["MedHouseVal"]

# Split data for regression
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# Train a baseline RandomForestRegressor
baseline_rf = RandomForestRegressor(random_state=42)
baseline_rf.fit(X_train_reg, y_train_reg)

# Evaluate the regression model
y_pred_reg = baseline_rf.predict(X_test_reg)
mse_reg = mean_squared_error(y_test_reg, y_pred_reg)
print("Baseline Regression MSE:", mse_reg)

# Save the regression model
with open("baseline_rf_model.pkl", "wb") as f:
    pickle.dump(baseline_rf, f)

# ----------------------------
# Classification Pipeline
# ----------------------------
# Bin the continuous target into discrete categories
df['MedHouseVal_bin'] = pd.cut(df['MedHouseVal'], bins=3, labels=["Low", "Medium", "High"])

# Define features and binned target for classification
X_cls = df.drop(["MedHouseVal", "MedHouseVal_bin"], axis=1)
y_cls = df["MedHouseVal_bin"]

# Split data for classification
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
    X_cls, y_cls, test_size=0.2, random_state=42
)

# Define a reduced hyperparameter grid for faster tuning
param_grid_clf = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10]
}

# Setup GridSearchCV for classifier with 3-fold CV and parallel processing
grid_search_clf = GridSearchCV(RandomForestClassifier(random_state=42),
                               param_grid_clf,
                               cv=3,
                               scoring='accuracy',
                               n_jobs=-1)
grid_search_clf.fit(X_train_cls, y_train_cls)

print("Best parameters (Classifier):", grid_search_clf.best_params_)
best_clf = grid_search_clf.best_estimator_

# Evaluate the tuned classifier
y_pred_cls = best_clf.predict(X_test_cls)
print("Tuned Classification Accuracy:", accuracy_score(y_test_cls, y_pred_cls))
print(classification_report(y_test_cls, y_pred_cls))

# Save the tuned classifier
with open("tuned_rf_classifier.pkl", "wb") as f:
    pickle.dump(best_clf, f)


# %%
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Define a parameter grid for the regressor
param_grid_reg = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20]
}

# Initialize a RandomForestRegressor
rf_reg = RandomForestRegressor(random_state=42)

# Setup GridSearchCV with 3-fold cross-validation and parallel processing
grid_search_reg = GridSearchCV(rf_reg, param_grid_reg, cv=3, 
                               scoring='neg_mean_squared_error', 
                               n_jobs=-1)

# Fit the grid search on the regression training data (ensure X_train_reg, y_train_reg are defined)
grid_search_reg.fit(X_train_reg, y_train_reg)

print("Best parameters (Regression):", grid_search_reg.best_params_)

# Retrieve the best estimator and evaluate on test set
best_rf_reg = grid_search_reg.best_estimator_
y_pred_reg = best_rf_reg.predict(X_test_reg)
mse_reg = mean_squared_error(y_test_reg, y_pred_reg)
print("Tuned Regression MSE:", mse_reg)


# %%
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Define a parameter grid for the classifier
param_grid_clf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20]
}

# Initialize a RandomForestClassifier
rf_clf = RandomForestClassifier(random_state=42)

# Setup GridSearchCV for the classifier with 3-fold cross-validation and parallel processing
grid_search_clf = GridSearchCV(rf_clf, param_grid_clf, cv=3, 
                               scoring='accuracy', 
                               n_jobs=-1)

# Fit the grid search on the classification training data (ensure X_train_cls, y_train_cls are defined)
grid_search_clf.fit(X_train_cls, y_train_cls)

print("Best parameters (Classification):", grid_search_clf.best_params_)

# Retrieve the best estimator and evaluate on test set
best_rf_clf = grid_search_clf.best_estimator_
y_pred_cls = best_rf_clf.predict(X_test_cls)
print("Tuned Classification Accuracy:", accuracy_score(y_test_cls, y_pred_cls))
print(classification_report(y_test_cls, y_pred_cls))


# %%