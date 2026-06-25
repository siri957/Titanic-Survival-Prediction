# =========================
# TITANIC SURVIVAL PREDICTION
# =========================

# 1. IMPORT LIBRARIES
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
    r2_score,
    confusion_matrix,
    classification_report
)

# 2. LOAD DATASET
df = pd.read_csv(r"C:\Users\SIRICHANDANA KOTA\Downloads\archive (2)\Titanic-Dataset.csv")

print("\n===== FIRST 5 ROWS =====")
print(df.head())

# 3. DATA EXPLORATION
print("\n===== INFO =====")
print(df.info())

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== STATISTICS =====")
print(df.describe())

# 4. DATA PREPROCESSING

# Fill missing values
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Drop Cabin (too many missing values)
df.drop(columns=['Cabin'], inplace=True)

# Encode categorical data
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# Drop unnecessary columns
df.drop(columns=['PassengerId', 'Name', 'Ticket'], inplace=True)

print("\n===== AFTER PREPROCESSING =====")
print(df.head())

# 5. SPLIT DATA
X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("\nTraining size:", X_train.shape)
print("Testing size:", X_test.shape)

# 6. MODEL TRAINING
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("\nModel training completed!")

# 7. PREDICTIONS
y_pred = model.predict(X_test)

print("\n===== SAMPLE PREDICTIONS =====")
print("Predicted:", y_pred[:10])
print("Actual   :", y_test.values[:10])

# 8. EVALUATION METRICS

accuracy = accuracy_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n===== MODEL EVALUATION =====")
print("Accuracy:", accuracy)
print("Mean Squared Error:", mse)
print("R2 Score:", r2)

print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(y_test, y_pred))

# 9. CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:\n", cm)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Titanic Survival Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
