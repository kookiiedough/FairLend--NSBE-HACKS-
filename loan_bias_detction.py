import pandas as pd
import numpy as np

# Seed for reproducibility
np.random.seed(42)

# Sample size
N = 1000

# Generate synthetic data
data = {
    'age': np.random.randint(18, 65, size=N),
    'income': np.random.randint(30000, 100000, size=N),
    'credit_score': np.random.randint(300, 850, size=N),
    'race': np.random.choice(['Black', 'Other'], size=N, p=[0.3, 0.7]),
    'loan_approved': np.random.randint(0, 2, size=N)
}

df = pd.DataFrame(data)

import matplotlib.pyplot as plt
import seaborn as sns

# Quick overview
print(df.describe())
print(df['race'].value_counts())

# Visualization of loan approval rates by race
sns.countplot(x='loan_approved', hue='race', data=df)
plt.title('Loan Approval by Race')
plt.show()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Prepare the data
X = df[['age', 'income', 'credit_score']]  # Features
y = df['loan_approved']  # Target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, predictions))

# Bias detection: Check approval rates by race in the test set
test_df = X_test.copy()
test_df['race'] = df.loc[test_df.index, 'race']
test_df['loan_approved'] = y_test
approval_rates = test_df.groupby('race')['loan_approved'].mean()
print(approval_rates)
# Mock mitigation: Adjust the decision threshold for demonstration purposes
# Note: In a real scenario, you would use a more sophisticated approach.

# Get probabilities instead of binary predictions
probabilities = model.predict_proba(X_test)[:, 1]  # Probability of being approved

# Adjusted threshold for demonstration (normally, you'd calculate this based on bias detection)
adjusted_threshold = 0.5  # Placeholder value

# Apply adjusted threshold to the Black group
test_df['predicted_approval'] = np.where((test_df['race'] == 'Black') & (probabilities > adjusted_threshold), 1, 0)

# Recalculate approval rates
new_approval_rates = test_df.groupby('race')['predicted_approval'].mean()
print(new_approval_rates)
