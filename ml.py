import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_decision_regions


# =========================
# 1. Load Dataset
# =========================

df = pd.read_csv('placement.csv')

print("First 5 rows:")
print(df.head())

print("\nDataset Information:")
df.info()

print("\nDataset Shape:")
print(df.shape)


# =========================
# 2. Remove Unnecessary Column
# =========================

df = df.iloc[:, 1:]

print("\nAfter removing first column:")
print(df.head())


# =========================
# 3. Visualization
# =========================

plt.scatter(
    df['cgpa'],
    df['iq'],
    c=df['placement']
)

plt.xlabel('CGPA')
plt.ylabel('IQ')
plt.title('CGPA vs IQ')
plt.show()


# =========================
# 4. Separate X and y
# =========================

X = df.iloc[:, 0:2]
y = df.iloc[:, -1]

print("\nFeatures (X):")
print(X)

print("\nTarget (y):")
print(y)

print("\nTarget shape:")
print(y.shape)


# =========================
# 5. Train-Test Split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.1,
    random_state=42
)

print("\nX_train:")
print(X_train)

print("\ny_train:")
print(y_train)

print("\nX_test:")
print(X_test)


# =========================
# 6. Feature Scaling
# =========================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nScaled X_train:")
print(X_train)

print("\nScaled X_test:")
print(X_test)


# =========================
# 7. Train Logistic Regression
# =========================

clf = LogisticRegression()

clf.fit(X_train, y_train)


# =========================
# 8. Prediction
# =========================

y_pred = clf.predict(X_test)

print("\nActual values:")
print(y_test.values)

print("\nPredicted values:")
print(y_pred)


# =========================
# 9. Accuracy
# =========================

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)


# =========================
# 10. Decision Boundary
# =========================

plot_decision_regions(
    X_train,
    y_train.values,
    clf=clf,
    legend=2
)

plt.xlabel('CGPA')
plt.ylabel('IQ')
plt.title('Logistic Regression Decision Boundary')
plt.show()


# =========================
# 11. Save Model
# =========================

pickle.dump(clf, open('model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))


print("\nModel saved successfully as model.pkl")