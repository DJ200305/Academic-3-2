# ============================================================
# Online 3 — Advanced Ensemble vs XGBoost
# ============================================================

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder,LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from scipy import stats

# ============================================================
# Part 0 — Data Loading
# ============================================================

# TODO: Load advanced_customer_data.csv
df = pd.read_csv('E:/3-2/CSE-330 ML Sessional/Online 3/advanced_customer_data.csv')
# TODO: Drop missing values
df = df.dropna(axis=0)
# TODO: Separate features (X) and target (y)
y = df['target']
X = df.drop(['target'],axis=1)
# TODO: Perform 80-20 train-test split (random_state=42)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,train_size=0.8,random_state=42)
# TODO: Identify numerical columns
num_cols = X.select_dtypes(['int64','float64']).columns
# TODO: Identify categorical columns
cat_cols = X.select_dtypes(['object']).columns
# TODO: Create ColumnTransformer:
#       - StandardScaler for numeric
#       - OneHotEncoder(handle_unknown='ignore') for categorical
column_transformer = ColumnTransformer(
    transformers=[
        ('num',StandardScaler(),num_cols),
        ('cat',OneHotEncoder(handle_unknown='ignore'),cat_cols)
    ]
)
# ============================================================
# Part 1 — Baseline Models
# ============================================================

# ---------------- Logistic Regression ----------------

# TODO: Create LogisticRegression (max_iter=1000)
model = LogisticRegression(max_iter=1000)
# TODO: Create Pipeline (preprocessor + LR)
pipeline = Pipeline(steps=[
    ('preprocessor',column_transformer),
    ('model',model)
])
# TODO: Train model
pipeline.fit(X_train,y_train)
# TODO: Compute training accuracy
y_train_pred = pipeline.predict(X_train)
train_acc = accuracy_score(y_train,y_train_pred)
# TODO: Compute test accuracy
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test,y_pred)
print("-----LR-----")
print(f"Training acc:{train_acc:.4f}")
print(f"Test acc:{acc:.4f}")
# ---------------- Feedforward Neural Network ----------------

# TODO: Create MLPClassifier
#       - At least 2 hidden layers
#       - max_iter >= 500
#       - random_state=42
mlpclf = MLPClassifier(
    hidden_layer_sizes=(64,32),
    max_iter=500,
    random_state=42
)
# TODO: Create Pipeline
pipeline1 = Pipeline(steps=[
    ('preprocessor',column_transformer),
    ('model',mlpclf)
])
# TODO: Train model
pipeline1.fit(X_train,y_train)
# TODO: Compute training accuracy
y_train_pred1 = pipeline1.predict(X_train)
train_acc1 = accuracy_score(y_train,y_train_pred1)
# TODO: Compute test accuracy
y_pred1 = pipeline1.predict(X_test)
acc1 = accuracy_score(y_test,y_pred1)
print("-----FNN-----")
print(f"Training Acc:{train_acc1:.4f}")
print(f"Test Acc:{acc1:.4f}")
# ============================================================
# Part 2 — Manual Bagging
# ============================================================

n_estimators = 10

# ---------------- Bagging Logistic Regression ----------------

# TODO:
# Create 10 bootstrap samples
# Train 10 LogisticRegression models
# Store models
models = []
for i in range(n_estimators):
    X_train1,y_train1 = resample(X_train,y_train,random_state=42+i)
    model = LogisticRegression(max_iter=1000,random_state=42+i)
    X_train1_xmod = column_transformer.fit_transform(X_train1)
    model.fit(X_train1_xmod,y_train)
    models.append(model)
# TODO:
# Perform majority voting (train + test)
all_pred_train = []
all_pred_test = []
X_train_mod = column_transformer.fit_transform(X_train)
X_test_mod = column_transformer.fit_transform(X_test)
for m in models:
    all_pred_train.append(m.predict(X_train_mod))
    all_pred_test.append(m.predict(X_test_mod))
# TODO:
# Compute bagging training accuracy
train_maj = stats.mode(all_pred_train)[0]
train_bagging_acc = accuracy_score(y_train,train_maj)
# TODO:
# Compute bagging test accuracy
test_maj = stats.mode(all_pred_test)[0]
test_bagging_acc = accuracy_score(y_test,test_maj)
print("------Bagging LR-------")
print(f"Training acc:{train_bagging_acc:.4f}")
print(f"Test acc:{test_bagging_acc:.4f}")
# ---------------- Bagging FNN ----------------

# TODO:
# Create 10 bootstrap samples
# Train 10 MLPClassifier models
# Store models
models1 = []
for i in range(n_estimators):
    X_train2,y_train2 = resample(X_train,y_train,random_state=42+i)
    model = MLPClassifier(
        hidden_layer_sizes=(64,32),
        max_iter=500,
        random_state=42+i
    )
    X_train2_xmod = column_transformer.fit_transform(X_train)
    model.fit(X_train2_xmod,y_train)
    models.append(model)
# TODO:
# Perform majority voting (train + test)
all_pred_train1 = []
all_pred_test1 = []
X_train_mod1 = column_transformer.fit_transform(X_train)
X_test_mod1 = column_transformer.fit_transform(X_test)
for m in models:
    all_pred_train1.append(m.predict(X_train_mod1))
    all_pred_test1.append(m.predict(X_test_mod1))
# TODO:
# Compute bagging training accuracy
train_maj1 = stats.mode(all_pred_train1)[0]
train_bagging_acc1 = accuracy_score(y_train,train_maj1)
# TODO:
# Compute bagging test accuracy
test_maj1 = stats.mode(all_pred_test1)[0]
test_bagging_acc1 = accuracy_score(y_test,test_maj1)
print("--------Bagging FNN--------")
print(f"Training acc:{train_bagging_acc1:.4f}")
print(f"Test acc:{test_bagging_acc1:.4f}")
# ============================================================
# Part 3 — XGBoost Hyperparameter Study
# ============================================================

n_estimators_list = [50, 200, 500]
max_depth_list = [3, 6, 10]
learning_rate_list = [0.01, 0.1, 0.3]
gamma_list = [0, 1, 5]
min_child_weight_list = [1, 5, 10]

# ---------------- XGBoost (Tree Booster) ----------------

# TODO:
# Loop over all combinations
# Train XGBClassifier with:
#       objective='binary:logistic'
#       booster='gbtree'
#       eval_metric='logloss'
#       random_state=42
best_acc = 0.0
worst_acc = 1.0
best_params = None
for i in n_estimators_list:
    for j in max_depth_list:
        for k in learning_rate_list:
            for l in gamma_list:
                for m in min_child_weight_list:
                    params = {
                        'n_estimators':i,
                        'max_depth':j,
                        'learning_rate':k,
                        'gamma':l,
                        'min_child_weight':m
                    }
                    X_train_xmod = column_transformer.fit_transform(X_train)
                    X_test_xmod = column_transformer.fit_transform(X_test)
                    clf = xgb.XGBClassifier(**params,
                        objective='binary:logistic',
                        booster='gbtree',
                        eval_metric='logloss',
                        random_state=42
                    )
                    clf.fit(X_train_xmod,y_train)
                    y_pred = clf.predict(X_test_xmod)
                    acc = accuracy_score(y_test,y_pred)
                    if worst_acc > acc:
                       worst_acc = acc
                    if best_acc <= acc:
                        best_acc = acc
                        best_params = params    
# TODO:
# Track:
#       best accuracy
#       worst accuracy
#       best params
#       worst params
print("--------XGBoost--------")
print(f"XGBoost Best Accuracy:{best_acc:.4f}")
print(f"XGBoost Worst Accuracy:{worst_acc:.4f}")
print(f"Best Params:{params}")

# ---------------- XGBoost (Linear Booster) ----------------

# TODO:
# Train XGBClassifier with booster='gblinear'

# TODO:
# Compute training accuracy

# TODO:
# Compute test accuracy


# ============================================================
# Part 4 — Stability Analysis
# ============================================================

# TODO:
# Repeat training of:
#   - Logistic Regression
#   - FNN
#   - Bagging LR
#   - Bagging FNN
#   - Best XGBoost
# For 5 different random seeds

# TODO:
# Compute mean accuracy for each model

# TODO:
# Compute standard deviation of accuracy


# ============================================================
# Part 5 — Overfitting Analysis
# ============================================================

# TODO:
# For best XGBoost configuration:
# Compute training accuracy

# TODO:
# Compute test accuracy

# TODO:
# Compute generalization gap


# ============================================================
# Part 6 — Feature Importance
# ============================================================

# TODO:
# Extract trained XGBoost (tree booster)

# TODO:
# Plot feature importance (importance_type='gain')

# TODO:
# Extract Logistic Regression coefficients

# TODO:
# Compare most important features


# ============================================================
# Part 7 — Model Persistence
# ============================================================

# TODO:
# Save best XGBoost model as JSON

# TODO:
# Load model

# TODO:
# Verify loaded model accuracy


# ============================================================
# Final Output
# ============================================================

# TODO:
# Print comparison table:

# Model | Train Acc | Test Acc | Mean Acc | Std Dev