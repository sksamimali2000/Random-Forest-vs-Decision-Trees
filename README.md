# 🚢 Titanic Survival Prediction - Data Preprocessing & Modeling

A complete data preprocessing pipeline and model training workflow on the **Titanic dataset**  
using **Decision Tree** and **Random Forest Classifier** from scikit-learn.

---

## 🚀 Project Overview

This project demonstrates:
- Data loading and splitting (Train/Test)  
- Data cleaning and feature engineering  
- Encoding categorical variables (`Sex`, `Embarked`)  
- Handling missing values with median imputation  
- Training **Decision Tree** and **Random Forest** classifiers  
- Evaluating model accuracy

---

## ⚡ Usage

```python
import numpy as np
import pandas as pd
from sklearn import model_selection, tree
from sklearn.ensemble import RandomForestClassifier

# Load dataset
titanic = pd.read_csv("titanic.csv")

# Train/test split
X = titanic.iloc[:, :-1]
Y = titanic.iloc[:, -1]
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, random_state=123211)

# Combine X_train and Y_train
X_Y_train = X_train.copy()
X_Y_train['Survived'] = Y_train
X_Y_train.to_csv('titanic_x_y_train.csv', index=False)
X_train.to_csv('titanic_x_train.csv', index=False)
Y_train.to_csv('titanic_y_train.csv', index=False)
X_test.to_csv('titanic_x_test.csv', index=False)
Y_test.to_csv('titanic_y_test.csv', index=False)

# Load and preprocess data
X_Y_train_l = pd.read_csv('titanic_x_y_train.csv')
X_test_l = pd.read_csv('titanic_x_test.csv')

# Drop unuseful columns
for col in ["Name", "Ticket", "Cabin", "Fare"]:
    del X_Y_train_l[col]
    del X_test_l[col]

# Fill missing values
X_Y_train_l = X_Y_train_l.fillna(X_Y_train_l.median())
X_test_l = X_test_l.fillna(X_test_l.median())

# Separate features and target
X_train_l = X_Y_train_l.iloc[:, :-1]
Y_train_l = X_Y_train_l.iloc[:, -1]

# Encode 'Sex' column
X_train_l['gender'] = X_train_l["Sex"].apply(lambda x: 1 if x == "male" else 2)
X_test_l['gender'] = X_test_l["Sex"].apply(lambda x: 1 if x == "male" else 2)
del X_train_l["Sex"]
del X_test_l["Sex"]

# Encode 'Embarked' column
def encode_embarked(val):
    return 1 if val == 'C' else 2 if val == 'Q' else 3

X_train_l['Embarked'] = X_train_l["Embarked"].apply(encode_embarked)
X_test_l['Embarked'] = X_test_l["Embarked"].apply(encode_embarked)

# Train Decision Tree Classifier
clf_tree = tree.DecisionTreeClassifier()
clf_tree.fit(X_train_l, Y_train_l)
print("Decision Tree Train Accuracy:", clf_tree.score(X_train_l, Y_train_l))
print("Decision Tree Test Accuracy:", clf_tree.score(X_test_l, Y_test))

# Train Random Forest Classifier
clf_rf = RandomForestClassifier(max_depth=6, random_state=0)
clf_rf.fit(X_train_l, Y_train_l)
print("Random Forest Train Accuracy:", clf_rf.score(X_train_l, Y_train_l))
print("Random Forest Test Accuracy:", clf_rf.score(X_test_l, Y_test))
```


✅ Key Outcomes

Data cleaned by removing irrelevant features

Missing values handled by median imputation

Categorical features encoded into numerical values

Both Decision Tree and Random Forest models trained successfully

Model performance evaluated on train and test sets

⚙️ Requirements

Python >= 3.7

pandas

numpy

scikit-learn

Install dependencies via:

pip install pandas numpy scikit-learn

📄 License

MIT License

Made with ❤️ by Sk Samim Ali
