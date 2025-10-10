from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000, random_state=42)


from sklearn.pipeline import Pipeline
from ohe_and_scaling import  X_train, y_train, X_test, preprocessor
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", model)
])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
