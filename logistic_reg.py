import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

train_data = pd.read_csv("data/ieee-fraud-detection/train_transaction.csv")
test_data = pd.read_csv("data/ieee-fraud-detection/test_transaction.csv")

numeric_columns = train_data.select_dtypes(include=['int64', 'float64']).columns
train_data[numeric_columns] = train_data[numeric_columns].fillna(train_data[numeric_columns].median())

categorical_columns = train_data.select_dtypes(include=['object']).columns
train_data = pd.get_dummies(train_data, columns=categorical_columns, drop_first=True)

X = train_data.drop(['isFraud'], axis=1)
y = train_data['isFraud']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)



model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)



y_pred = model.predict(X_val)
y_pred_prob = model.predict_proba(X_val)[:, 1]

print(classification_report(y_val, y_pred))

auc_score = roc_auc_score(y_val, y_pred_prob)
print(f"AUC-ROC Score: {auc_score:.4f}")


fpr, tpr, _ = roc_curve(y_val, y_pred_prob)
plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {auc_score:.4f})")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()