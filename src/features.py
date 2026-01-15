from sklearn.preprocessing import FunctionTransformer

def _clip(X):
    return X.clip(-3, 3)


def build_numeric_preprocess():
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clip", FunctionTransformer(_clip)),
        ]
    )