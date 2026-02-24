from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import pandas as pd
from xgboost import XGBClassifier
import optuna

# Load the dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standarized features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train baseline XGBoost model
baseline_model = XGBClassifier(eval_metric = 'logloss', random_state = 42)
baseline_model.fit(X_train,y_train)

# Evaluate the model
baseline_pred = baseline_model.predict(X_test)
accuracy_baseline = accuracy_score(y_test, baseline_pred)
print(f"BaseLine XGBoost Accuracy: {accuracy_baseline:.4f}")
print("ROC-AUC:", roc_auc_score(y_test, baseline_model.predict_proba(X_test)[:,1]))

# Define the objective fuction for Optuna
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.03),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10)
    }

    # Train XGBoost model with suggested params
    model = XGBClassifier(eval_metric='logloss',random_state=42, **params)
    model.fit(X_train,y_train)

    # Evaluate model on validation set
    roc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
    return roc

# Create an Optuna study
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=80)

# Best hyperparameters
print(f"Best hyperparameters: {study.best_params}")
print(f"Best ROC-AUC (Optuna): {study.best_value}")

# Train final model using Optuna best parameters
best_optuna_model = XGBClassifier(
    eval_metric='logloss',
    random_state=42,
    **study.best_params
)

best_optuna_model.fit(X_train, y_train)

# Evaluate Optuna tuned model
optuna_pred = best_optuna_model.predict(X_test)

print("\nOptuna Tuned Model Performance:")
print("Accuracy:", accuracy_score(y_test, optuna_pred))
print("ROC-AUC:",roc_auc_score(y_test,best_optuna_model.predict_proba(X_test)[:,1]))

# Define parameter grid
param_grid = {
    'n_estimators': [100,200,300],
    'max_depth': [3,5,7],
    'learning_rate': [0.01,0.1,0.2],
    'subsample': [0.6,0.8,1.0]
}

# Train XGBoost with grid search
grid_search = GridSearchCV(
    estimator=XGBClassifier(eval_metric='logloss',random_state=42),
    param_grid=param_grid,
    cv=3,
    scoring='accuracy',
    verbose=1
)

grid_search.fit(X_train,y_train)

# Best parameters
print(f"Grid Search best parameters: {grid_search.best_params_}")
print(f"Grid Search Best accuracy: {grid_search.best_score_}")

# Define parameter distribution
param_dist = {
    'n_estimators': [50,100,200,300,400],
    'max_depth': [3,5,7,9],
    'learning_rate': [0.01,0.05,0.1,0.2],
    'subsample': [0.6,0.7,0.8,0.9,1.0],
    'colsample_bytree':[0.6,0.7,0.8,0.9,1.0]
}

# Train the XGBoost with Random Search  
random_search = RandomizedSearchCV(
    estimator=XGBClassifier(eval_metric='logloss',random_state=42),
    param_distributions=param_dist,
    n_iter=50,
    scoring='accuracy',
    cv=3,
    verbose=1,
    random_state=42
)

random_search.fit(X_train,y_train)

# Best parameters and accuracy
print(f"\n\n\nRandom Search Best parameters: {random_search.best_params_}")
print(f"Random Search Accuracy: {random_search.best_score_}")
