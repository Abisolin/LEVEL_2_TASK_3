
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE
import joblib

df = pd.read_csv('creditcard.csv')
print("Shape:", df.shape)
print(df.head())

print(df.info())
print(df.isnull().sum())

print(df['Class'].value_counts())
sns.countplot(x='Class', data=df)
plt.title('Class Distribution (0=Not Fraud, 1=Fraud)')
plt.show()

best_model = models["Random Forest"]
joblib.dump(best_model, "fraud_model.pkl")
print("Model saved as fraud_model.pkl")

loaded_model = joblib.load("fraud_model.pkl")
sample = X_test.iloc[[0]]
pred = loaded_model.predict(sample)
print("Prediction for sample:", pred)
