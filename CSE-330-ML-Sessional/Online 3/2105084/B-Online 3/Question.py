import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb
import random
from scipy import stats

# -------------------- Reproducibility --------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

set_seed(42)

# -------------------- Load Dataset --------------------
# TODO: load your dataset 
df = pd.read_csv('chemical_quality_dataset.csv')

# -------------------- Separate Features and Target --------------------
# TODO: define feature matrix X and target vector y
# Example: target column is "quality"
y = df['quality']
X = df.drop(['quality'],axis=1)

# -------------------- Train-Test Split --------------------
# TODO: split data into train and test sets (80%-20%)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# -------------------- Scale Features (Optional) --------------------
# TODO: scale features if required
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)
# -------------------- Bagging with Decision Trees --------------------
n_estimators = 10
bagging_models = []

# TODO: train n_estimators Decision Tree models on bootstrap samples
# TODO: collect predictions from all models
# TODO: perform majority voting
# TODO: compute bagging accuracy (bagging_acc)
for i in range(n_estimators):
    X_train_mod,y_train_mod = resample(X_train_scaled,y_train,random_state=42+i)
    model = DecisionTreeClassifier(max_depth=3)
    model.fit(X_train_mod,y_train_mod)
    bagging_models.append(model)
bagging_pred = []
for m in bagging_models:
    m_pred = m.predict(X_test_scaled)
    acc = accuracy_score(y_test,m_pred)
    bagging_pred.append(acc)
bagging_acc = stats.mode(bagging_pred)[0]    
# -------------------- AdaBoost --------------------
# TODO: initialize a weak learner (DecisionTreeClassifier with max_depth=5)
# TODO: train AdaBoostClassifier 
# TODO: predict on test data and compute adaboost accuracy (adaboost_acc)
weak_dt = DecisionTreeClassifier(max_depth=5)
ab_c = AdaBoostClassifier(estimator=weak_dt)
ab_c.fit(X_train_scaled,y_train)
ab_pred = ab_c.predict(X_test_scaled)
adaboost_acc = accuracy_score(y_test,ab_pred)
# -------------------- XGBoost --------------------

# TODO: experiment XGBoost with different combinations of hyperparameters (n_estimators, max_depth, and learning_rate).

n_estimators_list = [5, 20, 50, 100]
max_depth_list = [1, 3, 5]
learning_rate_list = [0.01, 0.1, 0.3]

# TODO: train and predict with each combination
# Find the worst(Lowest possible XGBoost accuracy) and best(Highest possible XGBoost accuracy) XGBoost configurations
xgb_poor_acc = 1.0
xgb_best_acc = 0.0
for i in n_estimators_list:
    for j in max_depth_list:
        for k in learning_rate_list:
            params = {
                'n_estimators':i,
                'max_depth':j,
                'learning_rate':k
            }
            clf = xgb.XGBClassifier(**params)
            clf.fit(X_train_scaled,y_train)
            clf_pred = clf.predict(X_test_scaled)
            acc = accuracy_score(y_test,clf_pred)
            if xgb_poor_acc > acc:
                xgb_poor_acc = acc
            if xgb_best_acc <= acc:
                xgb_best_acc = acc    

# -------------------- Final Output --------------------
print("\n" + "="*60)
print("RESULTS:")
print("="*60)
print(f"Bagging Accuracy:               {round(bagging_acc, 4)}")
print(f"AdaBoost Accuracy:              {round(adaboost_acc, 4)}")
print("-"*60)
print(f"XGBoost (Worst Config) Accuracy:{round(xgb_poor_acc, 4)}")
print(f"XGBoost (Best Config) Accuracy: {round(xgb_best_acc, 4)}")
print("="*60)
