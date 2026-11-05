
import os
import time
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    average_precision_score
)

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

print("Loading datasets...")

train_df = pd.read_csv("dataset/Fraud_detection dataset/fraudTrain.csv")
test_df = pd.read_csv("dataset/Fraud_detection dataset/fraudTest.csv")

print("\nTraining Dataset Shape:", train_df.shape)
print("Testing Dataset Shape:", test_df.shape)

print("\nTraining Dataset Columns:")
print(train_df.columns.tolist())

print("\nFirst Five Records:")
print(train_df.head())

# Target column
target = "is_fraud"

if target not in train_df.columns:
    raise ValueError(
        "Target column 'is_fraud' not found. "
        "Please check your dataset column names."
    )

if target not in test_df.columns:
    raise ValueError(
        "Target column 'is_fraud' not found in test dataset."
    )

# Separate features and target
X_train = train_df.drop(columns=[target])
y_train = train_df[target]

X_test = test_df.drop(columns=[target])
y_test = test_df[target]

# Remove unnecessary identifier and personal detail columns
unnecessary_columns = [
    "Unnamed: 0",
    "trans_num",
    "cc_num",
    "first",
    "last",
    "street",
    "dob"
]

X_train = X_train.drop(
    columns=unnecessary_columns,
    errors="ignore"
)

X_test = X_test.drop(
    columns=unnecessary_columns,
    errors="ignore"
)

# Remove any remaining columns that could identify a person
# or are not useful for prediction
X_test = X_test.reindex(columns=X_train.columns)

# Identify numerical and categorical columns
numerical_columns = X_train.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_columns = X_train.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print("\nNumerical Columns:", numerical_columns)
print("\nCategorical Columns:", categorical_columns)

# Numerical preprocessing
numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical preprocessing
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Combine preprocessing
preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_columns),
    ("cat", categorical_pipeline, categorical_columns)
])

# Models
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=15,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=18,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}

results = []

best_model = None
best_model_name = None
best_f1 = -1

# Train and evaluate models
for name, classifier in models.items():

    print("\n" + "=" * 55)
    print("Training:", name)
    print("=" * 55)

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    start_time = time.time()

    model.fit(X_train, y_train)

    training_time = time.time() - start_time

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )
    pr_auc = average_precision_score(
        y_test, probabilities
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "PR AUC": pr_auc,
        "Training Time": training_time
    })

    print("Accuracy:", round(accuracy * 100, 2), "%")
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))
    print("PR AUC:", round(pr_auc, 4))
    print("Training Time:", round(training_time, 2), "seconds")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        zero_division=0
    ))

    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        best_model_name = name

# Model comparison
results_df = pd.DataFrame(results)

print("\n" + "=" * 55)
print("MODEL COMPARISON")
print("=" * 55)

print(results_df.to_string(index=False))

results_df.to_csv(
    "results/model_comparison.csv",
    index=False
)

# Save best model
joblib.dump(
    best_model,
    "models/best_fraud_detection_model.pkl"
)

print("\nBest Model:", best_model_name)

# Confusion matrix
best_predictions = best_model.predict(X_test)

cm = confusion_matrix(y_test, best_predictions)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Legitimate", "Fraud"],
    yticklabels=["Legitimate", "Fraud"]
)

plt.title("Fraud Detection - Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.tight_layout()

plt.savefig("results/confusion_matrix.png")
plt.close()

# Model comparison graph
plt.figure(figsize=(9, 5))

sns.barplot(
    data=results_df,
    x="Model",
    y="F1 Score"
)

plt.title("Model F1 Score Comparison")
plt.xticks(rotation=15)
plt.tight_layout()

plt.savefig("results/model_comparison.png")
plt.close()

print("\nProject completed successfully!")
print("Best model saved in models folder.")
print("Results saved in results folder.")

