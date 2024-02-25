import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve
from sklearn.utils.class_weight import compute_sample_weight
import matplotlib.pyplot as plt

# Seed for reproducibility
np.random.seed(42)

# Generate a synthetic dataset
data_size = 1000
data = {
    'Age': np.random.randint(18, 65, size=data_size),
    'Income': np.random.randint(30000, 100000, size=data_size),
    'CreditScore': np.random.randint(300, 850, size=data_size),
    'Race': np.random.choice(['Black', 'Not Black'], size=data_size, p=[0.3, 0.7]),
    'LoanApproval': np.random.choice([0, 1], size=data_size, p=[0.5, 0.5])
}

df = pd.DataFrame(data)

# Basic Data Analysis for Bias Detection
print(df.groupby('Race')['LoanApproval'].mean())

# Prepare data for training
X = df[['Age', 'Income', 'CreditScore']]  # Features
y = df['LoanApproval']  # Target variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions and initial accuracy
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Initial Model Accuracy: {accuracy}")

# --- Mitigation Strategy 1: Adjusting Decision Thresholds ---
probabilities = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, probabilities)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

adjusted_predictions = (probabilities >= optimal_threshold).astype(int)
adjusted_accuracy = accuracy_score(y_test, adjusted_predictions)
print(f"Adjusted Model Accuracy: {adjusted_accuracy}")

# --- Mitigation Strategy 2: Re-weighting Training Data ---
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
model_reweighted = LogisticRegression()
model_reweighted.fit(X_train, y_train, sample_weight=sample_weights)

reweighted_predictions = model_reweighted.predict(X_test)
reweighted_accuracy = accuracy_score(y_test, reweighted_predictions)
print(f"Reweighted Model Accuracy: {reweighted_accuracy}")

# Visualize the results
approvals_by_race = df.groupby('Race')['LoanApproval'].mean()
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.bar(approvals_by_race.index, approvals_by_race.values)
plt.title('Loan Approval Rates by Race')
plt.ylabel('Approval Rate')

plt.subplot(1, 2, 2)
accuracies = [accuracy, adjusted_accuracy, reweighted_accuracy]
labels = ['Initial', 'Adjusted Threshold', 'Reweighted']
plt.bar(labels, accuracies)
plt.title('Model Accuracies')
plt.ylabel('Accuracy')

plt.tight_layout()
plt.show()
