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
df = pd.read_csv('E:/3-2/CSE-330 ML Sessional/Online 3/C-Online 3/modified_large_iris_dataset_new.csv')


# TODO: separate features (X) and target (y)
# The species columns are one-hot encoded: species_0, species_1, species_2
# Convert them back to a single target column
df = df.dropna(axis=0)
X = df.drop(['species_0','species_1','species_2'],axis=1)
df['target'] = df[['species_0','species_1','species_2']].idxmax(axis=1).str[-1].astype(int)
y = df['target']


# TODO: train-test split (80%-20%)
X_train, X_test, y_train, y_test =train_test_split(X,y,test_size=0.2,train_size=0.8,random_state=42)


# TODO: scale the features if needed
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# -------------------- Single Logistic Regression --------------------

# TODO: train a single Logistic Regression (max_iter=1000, random_state=42) and compute lr_acc
lr_acc = 0.0
model = LogisticRegression(max_iter=1000,random_state=42)
model.fit(X_train_scaled,y_train)
y_pred = model.predict(X_test_scaled)
lr_acc = accuracy_score(y_test,y_pred)

# -------------------- Bagging with Logistic Regression --------------------

n_estimators = 12
models = []

# TODO: train n_estimators Logistic Regression (max_iter=1000, random_state=42) models on bootstrap samples
models = []
for i in range(n_estimators):
    X_train_scaled,y_train = resample(X_train_scaled,y_train,random_state=42+i)
    model = LogisticRegression(max_iter=1000,random_state=42+i)
    model.fit(X_train_scaled,y_train)
    models.append(model)


# TODO: collect predictions from all models
all_preds = []
for m in models:
    all_preds.append(m.predict(X_test_scaled))

# TODO: perform majority voting and compute bagging_acc
bagging_acc = []
for i in range(len(X_test_scaled)):
    pred = []
    for j in all_preds:
        y_pred = j[i]
        pred.append(y_pred)
    bagging_acc.append(stats.mode(pred)[0])    

bagging_acc = accuracy_score(y_test,bagging_acc)
# -------------------- XGBoost --------------------

# TODO: experiment XGBoost with different combinations of hyperparameters.
# Inspiration from typical XGBoost cheat-sheet tuning knobs.
n_estimators_list = [20, 80, 150]
max_depth_list = [2, 4, 6]
learning_rate_list = [0.03, 0.1, 0.2]
subsample_list = [0.8, 1.0]
colsample_bytree_list = [0.8, 1.0]
min_child_weight_list = [1, 5]
gamma_list = [0, 0.2]


# TODO: train and predict with each combination
# Find the worst (lowest) and best (highest) XGBoost accuracy + their configs
xgb_worst_acc = 1.0
xgb_best_acc = 0.0
xgb_worst_cfg = None
xgb_best_cfg = None

for i in n_estimators_list:
    for j in max_depth_list:
        for k in learning_rate_list:
            for l in subsample_list:
                for m in colsample_bytree_list:
                    for n in min_child_weight_list:
                        for o in gamma_list:
                            params = {
                                'n_estimators':i,
                                'max_depth':j,
                                'learning_rate':k,
                                'subsample':l,
                                'colsample_bytree':m,
                                'min_child_weight':n,
                                'gamma':o
                            }
                            model = xgb.XGBClassifier(**params)
                            model.fit(X_train_scaled,y_train)
                            y_pred = model.predict(X_test_scaled)
                            acc = accuracy_score(y_test,y_pred)
                            if acc <xgb_worst_acc:
                                xgb_worst_acc = acc
                                xgb_worst_cfg = params
                            if acc >= xgb_best_acc:
                                xgb_best_acc = acc
                                xgb_best_cfg = params


# -------------------- Final Output --------------------

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"Single Logistic Regression Accuracy: {round(lr_acc, 4)}")
print(f"Bagging (Logistic Regression) Accuracy: {round(bagging_acc, 4)}")
print("-" * 60)
print(f"XGBoost Worst Accuracy: {round(xgb_worst_acc, 4)}")
print(f"XGBoost Worst Config: {xgb_worst_cfg}")
print(f"XGBoost Best Accuracy: {round(xgb_best_acc, 4)}")
print(f"XGBoost Best Config: {xgb_best_cfg}")
print("=" * 60)
