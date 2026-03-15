import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import random

# -------------------- Reproducibility --------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

set_seed(42)

# -------------------- Step 1: Load Dataset --------------------
# TODO: load your dataset
df = pd.read_csv('E:/3-2/CSE-330 ML Sessional/Online 3/A-Online 3/diabetes.csv')

# -------------------- Step 2: Separate Features and Target --------------------
# TODO: define feature matrix X and target vector y
# Example: target column = "Outcome"
y = df['Outcome']
X = df.drop(['Outcome'],axis=1)

# -------------------- Step 3: Train-Test Split --------------------
# TODO: split data into training and test sets (80%-20%)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# -------------------- Step 4: Scale Features (Optional) --------------------
# TODO: scale features if required
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)
# --------------------Step 5: Stacking --------------------
# TODO: define base learners:
#   - DecisionTreeClassifier(max_depth=1, random_state=42)
#   - RandomForestClassifier(n_estimators=1, random_state=42)
#   - GradientBoostingClassifier(n_estimators=1, random_state=42)
dt = DecisionTreeClassifier(max_depth=1,random_state=42)
rf = RandomForestClassifier(n_estimators=1,random_state=42)
gb = GradientBoostingClassifier(n_estimators=1,random_state=42)
bases = [
    ('DT',dt),
    ('RF',rf),
    ('GB',gb)
]
# TODO: train base learners individually and compute their accuracies
dt.fit(X_train_scaled,y_train)
rf.fit(X_train_scaled,y_train)
gb.fit(X_train_scaled,y_train)
dt_pred = dt.predict(X_test_scaled)
rf_pred = dt.predict(X_test_scaled)
gb_pred = dt.predict(X_test_scaled)
dt_acc = accuracy_score(y_test,dt_pred)
rf_acc = accuracy_score(y_test,rf_pred)
gb_acc = accuracy_score(y_test,gb_pred)

base_accuracies = {
    'DT': dt_acc,
    'RF': rf_acc,
    'GB': gb_acc
}
# TODO: define meta-learner:
#   - LogisticRegression(max_iter=1000, random_state=42)
meta = LogisticRegression(max_iter=1000,random_state=42)
# TODO: train StackingClassifier on training data
st = StackingClassifier(estimators=bases,final_estimator=meta)
st.fit(X_train_scaled,y_train)
# TODO: predict on test data
st_pred = st.predict(X_test_scaled)
stacking_acc = accuracy_score(y_test,st_pred)
# TODO: compute stacking accuracy (stacking_acc)

# --------------------Step 6: XGBoost --------------------


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
            params = {
                'n_estimators':i,
                'max_depth':j,
                'learning_rate':k
            }
            clf = xgb.XGBClassifier(**params)
            clf.fit(X_train_scaled,y_train)
            clf_pred = clf.predict(X_test)
            clf_acc = accuracy_score(y_test,clf_pred)
            if xgb_poor_acc > clf_acc:
               xgb_poor_acc = clf_acc
            if xgb_best_acc <= clf_acc:
                xgb_best_acc = clf_acc   


# -------------------- Final Output --------------------
print("\n" + "="*60)
print("BASE LEARNER ACCURACIES:")
print("="*60)
for name, acc in base_accuracies.items():
    print(f"{name} Accuracy: {round(acc,4)}")
print("-"*60)
print(f"Stacking Accuracy: {round(stacking_acc,4)}")
print("-"*60)
print(f"XGBoost (Worst Config) Accuracy:{round(xgb_poor_acc,4)}")
print(f"XGBoost (Best Config) Accuracy: {round(xgb_best_acc,4)}")
print("="*60)
