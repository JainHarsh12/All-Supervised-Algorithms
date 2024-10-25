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

# Sample dataset
from sklearn.datasets import load_iris
data = load_iris()
X, y = data.data, data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

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

# Define parameter grids for hyperparameter tuning
param_grids = {
    'Logistic Regression': {'model__C': [0.1, 1.0, 10]},
    'k-NN': {'model__n_neighbors': [3, 5, 7]},
    'Decision Tree': {'model__max_depth': [3, 5, 7]},
    'Random Forest': {'model__n_estimators': [50, 100], 'model__max_depth': [5, 10]},
    'SVM': {'model__C': [0.1, 1, 10], 'model__kernel': ['linear', 'rbf']},
    'Naive Bayes': {},  # No hyperparameters for GaussianNB
    'Gradient Boosting': {'model__n_estimators': [50, 100], 'model__learning_rate': [0.1, 0.01]},
    'AdaBoost': {'model__n_estimators': [50, 100], 'model__learning_rate': [0.1, 0.01]}
}

# Create a list to store results
results = {}

# Loop through each model and perform hyperparameter tuning
for model_name, model in models.items():
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    
    # Apply GridSearchCV
    grid_search = GridSearchCV(pipeline, param_grid=param_grids[model_name], cv=3, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    # Best model evaluation
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Save the results
    results[model_name] = {
        'Best Parameters': grid_search.best_params_,
        'Accuracy': accuracy
    }

# Display results
for model_name, result in results.items():
    print(f"Model: {model_name}")
    print(f"Best Parameters: {result['Best Parameters']}")
    print(f"Accuracy: {result['Accuracy']:.2f}")
    print("-" * 30)
