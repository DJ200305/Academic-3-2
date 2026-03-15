import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import xgboost as xgb
import random
from scipy import stats

# Reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

set_seed(42)

# TODO: load the dataset
df = pd.read_csv('E:/3-2/CSE-330 ML Sessional/Online 3/C-Online 3/modified_large_iris_dataset.csv')
df = df.dropna(axis=0)


# TODO: separate features (X) and target (y)
# The species columns are one-hot encoded: species_0, species_1, species_2
# Convert them back to a single target column
df['target'] = df[['species_0','species_1','species_2']].idxmax(axis=1).str[-1].astype(int)
X = df.drop(['target','species_0','species_1','species_2'],axis=1)
y = df['target']


# TODO: train-test split (80%-20%)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,train_size=0.8,random_state=42)



# TODO: scale the features if needed
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)




# -------------------- Single Logistic Regression --------------------

# TODO: train a single Logistic Regression (max_iter=1000, random_state=42) and compute lr_acc
model = LogisticRegression(max_iter=1000,random_state=42)
model.fit(X_train_scaled,y_train)
y_pred = model.predict(X_test_scaled)
lr_acc = accuracy_score(y_test,y_pred)



# -------------------- Bagging with Logistic Regression --------------------

n_estimators = 10
models = []

# TODO: train n_estimators Logistic Regression (max_iter=1000, random_state=42) models on bootstrap samples
params = {
    'max_iter':1000,
    'random_state':42
}
for i in range(n_estimators):
    X_b,y_b = resample(X_train_scaled,y_train,replace=True,random_state=42+i)
    model = LogisticRegression(**params)
    model.fit(X_b,y_b)
    models.append(model)



# TODO: collect predictions from all models
all = []
for m in models:
    all.append(m.predict(X_test_scaled))



# TODO: perform majority voting and compute bagging_acc
bagging_acc = []
for i in range(len(X_test_scaled)):
    pred = []
    for m in all:
        pred.append(m[i])
    bagging_acc.append(stats.mode(pred,keepdims=True)[0][0])    

bagging_acc = accuracy_score(y_test,bagging_acc)


# -------------------- XGBoost --------------------

# TODO: experiment XGBoost with different combinations of hyperparameters (n_estimators, max_depth, and learning_rate).

n_estimators_list = [5, 20, 50]
max_depth_list = [1, 3, 5]
learning_rate_list = [0.01, 0.1, 0.3]



# TODO: train and predict with each combination
# Find the worst(Lowest possible XGBoost accuracy) and best(Highest possible XGBoost accuracy) XGBoost configurations

xgb_poor_acc = 1.0
xgb_best_acc = 0.0
for i in n_estimators_list:
    for j in max_depth_list:
        for k in learning_rate_list:
            model = xgb.XGBClassifier(
                n_estimators=i,
                max_depth=j,
                learning_rate=k
            )
            model.fit(X_train_scaled,y_train)
            pred = model.predict(X_test_scaled)
            acc = accuracy_score(y_test,pred)
            if acc < xgb_poor_acc:
                xgb_poor_acc = acc
            if acc >= xgb_best_acc:
                xgb_best_acc = acc    


# -------------------- Final Output --------------------

print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"Single Logistic Regression Accuracy: {round(lr_acc, 4)}")
print(f"Bagging (Logistic Regression) Accuracy: {round(bagging_acc, 4)}")
print("-" * 60)
print(f"XGBoost Worst Accuracy: {round(xgb_poor_acc, 4)}")
print(f"XGBoost Best Accuracy: {round(xgb_best_acc, 4)}")
print("=" * 60)