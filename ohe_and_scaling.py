from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
cat_cols = ["checking_account", "credit_history", "purpose", "savings_account",
            "employment_since", "personal_status_sex", "other_debtors",
            "property", "other_installment_plans", "housing",
            "job", "telephone", "foreign_worker"]
num_cols = ["duration", "credit_amount", "installment_rate", "residence_since",
            "age", "number_credits", "dependents"]
y = df["target"].map({1: 0, 2: 1})

X = df.drop("target", axis=1)

#split:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
#preprocessor:

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(drop="first"), cat_cols)
    ]
)