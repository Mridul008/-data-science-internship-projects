import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# TITANIC SURVIVAL PREDICTION USING RANDOM FOREST
# ============================================================


# ------------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------------

file_path = "titanic.csv"

data = pd.read_csv(
    file_path
)

print(
    "Titanic dataset loaded successfully."
)


# ------------------------------------------------------------
# 2. FIRST 5 ROWS
# ------------------------------------------------------------

print("\n==============================")
print("FIRST 5 ROWS")
print("==============================")

print(
    data.head()
)


# ------------------------------------------------------------
# 3. DATASET SHAPE
# ------------------------------------------------------------

print("\n==============================")
print("DATASET SHAPE")
print("==============================")

print(
    "Rows:",
    data.shape[0]
)

print(
    "Columns:",
    data.shape[1]
)


# ------------------------------------------------------------
# 4. COLUMN NAMES
# ------------------------------------------------------------

print("\n==============================")
print("COLUMN NAMES")
print("==============================")

print(
    data.columns.tolist()
)


# ------------------------------------------------------------
# 5. DATA TYPES
# ------------------------------------------------------------

print("\n==============================")
print("DATA TYPES")
print("==============================")

print(
    data.dtypes
)


# ------------------------------------------------------------
# 6. MISSING VALUES
# ------------------------------------------------------------

print("\n==============================")
print("MISSING VALUES")
print("==============================")

print(
    data.isnull().sum()
)


# ------------------------------------------------------------
# 7. DUPLICATE RECORDS
# ------------------------------------------------------------

print("\n==============================")
print("DUPLICATE RECORDS")
print("==============================")

print(
    "Duplicate rows:",
    data.duplicated().sum()
)


# ------------------------------------------------------------
# 8. STATISTICAL SUMMARY
# ------------------------------------------------------------

print("\n==============================")
print("STATISTICAL SUMMARY")
print("==============================")

print(
    data.describe()
)


# ------------------------------------------------------------
# 9. SURVIVAL COUNT
# ------------------------------------------------------------

print("\n==============================")
print("SURVIVAL COUNT")
print("==============================")

print(
    data["Survived"].value_counts()
)


# ------------------------------------------------------------
# 10. SURVIVAL PERCENTAGE
# ------------------------------------------------------------

survival_percentage = (
    data["Survived"]
    .value_counts(
        normalize=True
    )
    * 100
)

print("\n==============================")
print("SURVIVAL PERCENTAGE")
print("==============================")

print(
    survival_percentage
)


# ------------------------------------------------------------
# 11. SURVIVAL COUNT GRAPH
# ------------------------------------------------------------

plt.figure(
    figsize=(7, 5)
)

sns.countplot(
    data=data,
    x="Survived"
)

plt.title(
    "Titanic Survival Count"
)

plt.xlabel(
    "Survived (0 = No, 1 = Yes)"
)

plt.ylabel(
    "Number of Passengers"
)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 12. SURVIVAL BY GENDER
# ------------------------------------------------------------

plt.figure(
    figsize=(7, 5)
)

sns.countplot(
    data=data,
    x="Sex",
    hue="Survived"
)

plt.title(
    "Survival by Gender"
)

plt.xlabel(
    "Gender"
)

plt.ylabel(
    "Number of Passengers"
)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 13. SURVIVAL BY PASSENGER CLASS
# ------------------------------------------------------------

plt.figure(
    figsize=(7, 5)
)

sns.countplot(
    data=data,
    x="Pclass",
    hue="Survived"
)

plt.title(
    "Survival by Passenger Class"
)

plt.xlabel(
    "Passenger Class"
)

plt.ylabel(
    "Number of Passengers"
)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 14. AGE DISTRIBUTION
# ------------------------------------------------------------

plt.figure(
    figsize=(8, 5)
)

sns.histplot(
    data["Age"].dropna(),
    bins=30,
    kde=True
)

plt.title(
    "Age Distribution of Passengers"
)

plt.xlabel(
    "Age"
)

plt.ylabel(
    "Number of Passengers"
)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 15. AGE DISTRIBUTION BY SURVIVAL
# ------------------------------------------------------------

plt.figure(
    figsize=(8, 5)
)

sns.boxplot(
    data=data,
    x="Survived",
    y="Age"
)

plt.title(
    "Age Distribution by Survival"
)

plt.xlabel(
    "Survived"
)

plt.ylabel(
    "Age"
)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 16. FARE DISTRIBUTION
# ------------------------------------------------------------

plt.figure(
    figsize=(8, 5)
)

sns.histplot(
    data["Fare"],
    bins=30,
    kde=True
)

plt.title(
    "Fare Distribution"
)

plt.xlabel(
    "Fare"
)

plt.ylabel(
    "Frequency"
)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 17. SURVIVAL RATE BY CLASS
# ------------------------------------------------------------

survival_by_class = (
    data.groupby(
        "Pclass"
    )["Survived"]
    .mean()
    * 100
)

print("\n==============================")
print("SURVIVAL RATE BY CLASS")
print("==============================")

print(
    survival_by_class
)


# ------------------------------------------------------------
# 18. SURVIVAL RATE BY CLASS GRAPH
# ------------------------------------------------------------

plt.figure(
    figsize=(7, 5)
)

survival_by_class.plot(
    kind="bar"
)

plt.title(
    "Survival Rate by Passenger Class"
)

plt.xlabel(
    "Passenger Class"
)

plt.ylabel(
    "Survival Rate (%)"
)

plt.xticks(
    rotation=0
)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 19. SURVIVAL RATE BY GENDER
# ------------------------------------------------------------

survival_by_gender = (
    data.groupby(
        "Sex"
    )["Survived"]
    .mean()
    * 100
)

print("\n==============================")
print("SURVIVAL RATE BY GENDER")
print("==============================")

print(
    survival_by_gender
)


# ------------------------------------------------------------
# 20. SURVIVAL RATE BY GENDER GRAPH
# ------------------------------------------------------------

plt.figure(
    figsize=(7, 5)
)

survival_by_gender.plot(
    kind="bar"
)

plt.title(
    "Survival Rate by Gender"
)

plt.xlabel(
    "Gender"
)

plt.ylabel(
    "Survival Rate (%)"
)

plt.xticks(
    rotation=0
)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 21. SELECT FEATURES AND TARGET
# ------------------------------------------------------------

features = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare"
]

target = "Survived"

X = data[features]

y = data[target]


# ------------------------------------------------------------
# 22. DEFINE NUMERICAL AND CATEGORICAL FEATURES
# ------------------------------------------------------------

numerical_columns = [
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare"
]

categorical_columns = [
    "Sex"
]


# ------------------------------------------------------------
# 23. NUMERICAL PREPROCESSING
# ------------------------------------------------------------

numerical_pipeline = Pipeline([

    (
        "imputer",
        SimpleImputer(
            strategy="median"
        )
    )

])


# ------------------------------------------------------------
# 24. CATEGORICAL PREPROCESSING
# ------------------------------------------------------------

categorical_pipeline = Pipeline([

    (
        "imputer",
        SimpleImputer(
            strategy="most_frequent"
        )
    ),

    (
        "encoder",
        OneHotEncoder(
            handle_unknown="ignore"
        )
    )

])


# ------------------------------------------------------------
# 25. COMBINE PREPROCESSING
# ------------------------------------------------------------

preprocessor = ColumnTransformer([

    (
        "numerical",
        numerical_pipeline,
        numerical_columns
    ),

    (
        "categorical",
        categorical_pipeline,
        categorical_columns
    )

])


# ------------------------------------------------------------
# 26. RANDOM FOREST MODEL
# ------------------------------------------------------------

model = Pipeline([

    (
        "preprocessor",
        preprocessor
    ),

    (
        "classifier",
        RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
    )

])


# ------------------------------------------------------------
# 27. TRAIN / TEST SPLIT
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
)


# ------------------------------------------------------------
# 28. DATA SPLIT INFORMATION
# ------------------------------------------------------------

print("\n==============================")
print("DATA SPLIT")
print("==============================")

print(
    "Training samples:",
    len(X_train)
)

print(
    "Testing samples:",
    len(X_test)
)


# ------------------------------------------------------------
# 29. TRAIN MODEL
# ------------------------------------------------------------

model.fit(
    X_train,
    y_train
)

print(
    "\nModel training completed."
)


# ------------------------------------------------------------
# 30. MAKE PREDICTIONS
# ------------------------------------------------------------

y_pred = model.predict(
    X_test
)


# ------------------------------------------------------------
# 31. CALCULATE PERFORMANCE METRICS
# ------------------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)


# ------------------------------------------------------------
# 32. MODEL PERFORMANCE
# ------------------------------------------------------------

print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print(
    "Accuracy:",
    round(
        accuracy * 100,
        2
    ),
    "%"
)

print(
    "Precision:",
    round(
        precision,
        3
    )
)

print(
    "Recall:",
    round(
        recall,
        3
    )
)

print(
    "F1-score:",
    round(
        f1,
        3
    )
)


# ------------------------------------------------------------
# 33. CLASSIFICATION REPORT
# ------------------------------------------------------------

print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# ------------------------------------------------------------
# 34. CONFUSION MATRIX
# ------------------------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\n==============================")
print("CONFUSION MATRIX")
print("==============================")

print(
    cm
)


# ------------------------------------------------------------
# 35. CONFUSION MATRIX GRAPH
# ------------------------------------------------------------

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Did Not Survive",
        "Survived"
    ]
)

disp.plot()

plt.title(
    "Titanic Survival Classification"
)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 36. PROJECT SUMMARY
# ------------------------------------------------------------

print("\n==============================")
print("PROJECT SUMMARY")
print("==============================")

print(
    "Total passengers:",
    len(data)
)

print(
    "Training passengers:",
    len(X_train)
)

print(
    "Testing passengers:",
    len(X_test)
)

print(
    "Model accuracy:",
    round(
        accuracy * 100,
        2
    ),
    "%"
)

print(
    "\nTitanic Survival Analysis "
    "completed successfully."
)
