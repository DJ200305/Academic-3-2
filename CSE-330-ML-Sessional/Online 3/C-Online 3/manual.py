import numpy as np
import random

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# =========================================================
# Reproducibility
# =========================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

set_seed(42)


# =========================================================
# Dataset (Binary for Boosting Simplicity)
# =========================================================

X, y = make_classification(
    n_samples=1200,
    n_features=10,
    n_informative=6,
    n_classes=2,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

scaler = StandardScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test  = scaler.transform(X_test)


# =========================================================
# -------------------- BAGGING ----------------------------
# =========================================================

n_estimators = 10
n_samples = len(scaled_X_train)

bag_models = []

for i in range(n_estimators):
    idx = np.random.choice(n_samples, n_samples, replace=True)
    X_boot = scaled_X_train[idx]
    y_boot = y_train[idx]

    model = DecisionTreeClassifier(max_depth=3, random_state=42+i)
    model.fit(X_boot, y_boot)
    bag_models.append(model)

# Majority vote
bag_preds = np.array([m.predict(scaled_X_test) for m in bag_models])
final_bag_pred = []

for i in range(len(scaled_X_test)):
    values, counts = np.unique(bag_preds[:, i], return_counts=True)
    final_bag_pred.append(values[np.argmax(counts)])

bagging_acc = accuracy_score(y_test, final_bag_pred)


# =========================================================
# -------------------- ADABOOST ---------------------------
# =========================================================

n_estimators = 15
n_samples = len(scaled_X_train)

sample_weights = np.ones(n_samples) / n_samples

boost_models = []
boost_alphas = []

for i in range(n_estimators):

    model = DecisionTreeClassifier(max_depth=1, random_state=42+i)
    model.fit(scaled_X_train, y_train, sample_weight=sample_weights)

    preds = model.predict(scaled_X_train)

    incorrect = (preds != y_train).astype(int)
    error = np.dot(sample_weights, incorrect)
    error = np.clip(error, 1e-10, 1 - 1e-10)

    alpha = 0.5 * np.log((1 - error) / error)

    y_signed = np.where(y_train == 1, 1, -1)
    p_signed = np.where(preds == 1, 1, -1)

    sample_weights *= np.exp(-alpha * y_signed * p_signed)
    sample_weights /= sample_weights.sum()

    boost_models.append(model)
    boost_alphas.append(alpha)

# Weighted vote
boost_preds = np.array([m.predict(scaled_X_test) for m in boost_models])
boost_preds_signed = np.where(boost_preds == 1, 1, -1)

alphas_arr = np.array(boost_alphas).reshape(-1, 1)
weighted_votes = np.sum(alphas_arr * boost_preds_signed, axis=0)

final_boost_pred = (weighted_votes >= 0).astype(int)
boost_acc = accuracy_score(y_test, final_boost_pred)


# =========================================================
# -------------------- STACKING ---------------------------
# =========================================================

# Base models
base_models = [
    LogisticRegression(max_iter=1000, random_state=42),
    DecisionTreeClassifier(max_depth=4, random_state=42)
]

meta_model = LogisticRegression(max_iter=1000, random_state=42)

# Train base models
train_meta_features = []
test_meta_features  = []

for model in base_models:
    model.fit(scaled_X_train, y_train)

    train_meta_features.append(model.predict(scaled_X_train))
    test_meta_features.append(model.predict(scaled_X_test))

train_meta_X = np.column_stack(train_meta_features)
test_meta_X  = np.column_stack(test_meta_features)

# Train meta learner
meta_model.fit(train_meta_X, y_train)

stack_pred = meta_model.predict(test_meta_X)
stack_acc = accuracy_score(y_test, stack_pred)


# =========================================================
# -------------------- RESULTS ----------------------------
# =========================================================

print("="*50)
print("Manual Ensemble Comparison")
print("="*50)
print(f"Bagging Accuracy  : {bagging_acc:.4f}")
print(f"AdaBoost Accuracy : {boost_acc:.4f}")
print(f"Stacking Accuracy : {stack_acc:.4f}")
print("="*50)