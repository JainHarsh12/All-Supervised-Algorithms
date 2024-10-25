import pytest
import numpy as np  # Add this import for np to be recognized
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Define models
models = {
    'Logistic Regression': LogisticRegression(),
    'k-NN': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(),
    'Naive Bayes': GaussianNB(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'AdaBoost': AdaBoostClassifier()
}

# Parameter grids for hyperparameter tuning
param_grids = {
    'Logistic Regression': {'model__C': [0.1, 1.0]},
    'k-NN': {'model__n_neighbors': [3, 5]},
    'Decision Tree': {'model__max_depth': [3, 5]},
    'Random Forest': {'model__n_estimators': [50], 'model__max_depth': [5]},
    'SVM': {'model__C': [0.1, 1], 'model__kernel': ['linear']},
    'Naive Bayes': {},
    'Gradient Boosting': {'model__n_estimators': [50], 'model__learning_rate': [0.1]},
    'AdaBoost': {'model__n_estimators': [50], 'model__learning_rate': [0.1]}
}

# Fixture to prepare data
@pytest.fixture
def data():
    dataset = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

# Test if each model runs successfully and returns expected output
@pytest.mark.parametrize("model_name", models.keys())
def test_model_pipeline(model_name, data):
    X_train, X_test, y_train, y_test = data
    model = models[model_name]
    param_grid = param_grids[model_name]
    
    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    
    # GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=3, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    # Check best model and predict
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    # Assertions
    assert isinstance(y_pred, (list, tuple, np.ndarray)), "Prediction output should be an array or list."
    accuracy = accuracy_score(y_test, y_pred)
    assert 0 <= accuracy <= 1, "Accuracy should be within [0, 1]."
    assert accuracy > 0.5, f"Model {model_name} should perform better than random guessing."

    print(f"Model: {model_name}, Best Parameters: {grid_search.best_params_}, Accuracy: {accuracy:.2f}")
