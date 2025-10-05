from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def grid_search_logreg(X_train, y_train):
    param_grid = {
        "C": [0.001, 0.01, 0.1, 1, 10, 100],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear"]
    }

    log_reg = LogisticRegression(random_state=42, max_iter=1000)

    gs = GridSearchCV(
        estimator=log_reg,
        param_grid=param_grid,
        scoring="accuracy",
        cv=5,
        n_jobs=-1
    )

    gs.fit(X_train, y_train)

    print("Best parameters:", gs.best_params_)
    print("Best CV accuracy:", gs.best_score_)

    return gs.best_estimator_

def grid_search_rf(X_train, y_train):
    param_grid = {
        "n_estimators": [100, 300, 500],
        "max_depth": [None, 5, 10, 20],
        "max_features": ["sqrt", "log2"],
        "class_weight": ["balanced"]  # helps with imbalance
    }

    gs = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        scoring="accuracy",
        cv=5,
        n_jobs=-1
    )
    gs.fit(X_train, y_train)
    print("Best parameters:", gs.best_params_)
    print("Best CV accuracy:", gs.best_score_)
    return gs.best_estimator_