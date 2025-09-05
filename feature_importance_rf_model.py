from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import numpy as np
import pandas as pd
import seaborn as sns

def FeatureImportanceRfModel(X_train, y_train, X_test,feat_labels, threshold="median"):
    """
    Parameters
    ----------
    X_train : pd.DataFrame
        Training data (features).
    y_train : pd.Series or np.array
        Target variable.
    feat_labels : list
        List of feature names (column labels).
    threshold : str or float, default="median"
        Feature selection threshold in SelectFromModel.
        Options: "median", "mean" or float (e.g., 0.05).

    Returns
    -------
    feat_importances : pd.DataFrame
        Sorted table with features and their importance scores.
    X_selected : np.array
        Training data reduced to selected features.
    """

    forest = RandomForestClassifier(n_estimators=500, random_state=1)
    forest.fit(X_train, y_train)

    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]  # sort descending

    feat_importances = pd.DataFrame({
        "Feature": np.array(feat_labels)[indices],
        "Importance": importances[indices]
    })


    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=feat_importances)
    plt.title("Feature Importances (Random Forest)", fontsize=14)
    plt.tight_layout()
    plt.show()

    sfm = SelectFromModel(forest, threshold=threshold, prefit=True)
    X_selected = sfm.transform(X_train)
    X_test_selected = sfm.transform(X_test)
    print(f"Number of selected features: {X_selected.shape[1]}")

    return feat_importances, X_selected, X_test_selected
