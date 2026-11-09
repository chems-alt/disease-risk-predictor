"""Feature definitions shared by training and the API."""
import numpy as np

FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
TARGET = "Outcome"

# In this dataset a value of 0 means "not measured" for these columns
# (a living person cannot have 0 glucose or 0 BMI).
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
_ZERO_IDX = [FEATURES.index(c) for c in ZERO_AS_MISSING]


def zeros_to_nan(X):
    """Replace impossible zeros by NaN so the imputer can handle them."""
    X = np.array(X, dtype=float, copy=True)
    for i in _ZERO_IDX:
        col = X[:, i]
        col[col == 0] = np.nan
    return X
