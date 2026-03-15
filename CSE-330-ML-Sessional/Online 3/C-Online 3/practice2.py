import numpy as np
import pandas as pd
import random
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import VotingClassifier



# -------------------- Reproducibility --------------------

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

set_seed(42)



# -------------------- Load Dataset --------------------

# TODO: Load stacking_dataset.csv
df = pd.read_csv('E:/3-2/CSE-330 ML Sessional/Online 3/stacking_dataset.csv')
# TODO: Drop missing values if any
df = df.dropna(axis=0)


# -------------------- Target Reconstruction --------------------

# TODO: Convert one-hot target columns (class_0, class_1, class_2)
#       back into single integer target column
df['target'] = df[['class_0','class_1','class_2']].idxmax(axis=1).str[-1].astype(int)
y = df['target']
X = df.drop(['target','class_0','class_1','class_2'],axis=1)
# TODO: Separate X and y



# -------------------- Train-Test Split --------------------

# TODO: Perform 75%-25% split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,train_size=0.75,random_state=42)


# -------------------- Scaling --------------------

# TODO: Apply StandardScaler on training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)
# TODO: Transform test data



# =========================================================
# -------------------- Base Models -----------------------
# =========================================================

# TODO: Train single Logistic Regression
lr = LogisticRegression(max_iter=1000,random_state=42)
# TODO: Compute its accuracy
lr.fit(X_train_scaled,y_train)
lr_pred = lr.predict(X_test_scaled)
lr_acc = accuracy_score(y_test,lr_pred)


# TODO: Train single Random Forest
rf = RandomForestClassifier(n_estimators=10,random_state=42)
# TODO: Compute its accuracy
rf.fit(X_train_scaled,y_train)
rf_pred = rf.predict(X_test_scaled)
rf_acc = accuracy_score(y_test,rf_pred)


# TODO: Train single MLP (2 hidden layers)
mlp = MLPClassifier(hidden_layer_sizes=(64,32),random_state=42)
# TODO: Compute its accuracy
mlp.fit(X_train_scaled,y_train)
mlp_pred = mlp.predict(X_test_scaled)
mlp_acc = accuracy_score(y_test,mlp_pred)


# =========================================================
# -------------------- Hard Voting -----------------------
# =========================================================

# TODO: Create VotingClassifier using LR, RF, MLP
vm = VotingClassifier(estimators=[
    ('lr',lr),
    ('rf',rf),
    ('mlp',mlp)
])
# TODO: Train voting model
vm.fit(X_train_scaled,y_train)
# TODO: Compute voting accuracy
vm_pred = vm.predict(X_test_scaled)
vm_acc = accuracy_score(y_test,vm_pred)


# =========================================================
# -------------------- Stacking --------------------------
# =========================================================

# TODO: Define base learners (LR, RF, MLP)
bases = [('lr',lr),('rf',rf),('mlp',mlp)]
# TODO: Define meta learner (LogisticRegression)
meta = LogisticRegression(max_iter=2000)
# TODO: Build StackingClassifier
st = StackingClassifier(estimators=bases,final_estimator=meta)
# TODO: Train stacking model
st.fit(X_train_scaled,y_train)
# TODO: Compute stacking accuracy
st_pred = st.predict(X_test_scaled)
st_acc = accuracy_score(y_test,st_pred)


# =========================================================
# -------------------- XGBoost Grid ----------------------
# =========================================================

# TODO: Define hyperparameter lists:
#       n_estimators_list
#       max_depth_list
#       learning_rate_list

# TODO: Iterate through all combinations
# TODO: Train model for each configuration
# TODO: Track best accuracy
# TODO: Track worst accuracy



# =========================================================
# -------------------- Final Output ----------------------
# =========================================================

print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"Stacking acc:{st_acc:.4f}")
print(f"Hard Voting acc:{vm_acc:.4f}")
# TODO: Print all accuracies:
#       LR
#       RF
#       MLP
#       Voting
#       Stacking
#       XGBoost Best
#       XGBoost Worst

print("="*60)