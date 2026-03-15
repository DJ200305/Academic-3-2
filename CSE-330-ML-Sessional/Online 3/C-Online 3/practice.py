import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# TODO: Load customer_data.csv
df = pd.read_csv('E:/3-2/CSE-330 ML Sessional/Online 3/customer_data.csv')
# TODO: Drop missing values (if any)
df = df.dropna(axis=0)
# TODO: Separate features (X) and target (y)
y = df['purchased']
X = df.drop(['purchased'],axis=1)
# TODO: Perform 80-20 train-test split with random_state=42
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,train_size=0.8,random_state=42)
# TODO: Identify numerical columns
num_cols = X.select_dtypes(include=['int64','float64']).columns
# TODO: Identify categorical columns
cat_cols = X.select_dtypes(include=['object']).columns
# TODO: Create ColumnTransformer:
#       - Apply StandardScaler to numerical features
#       - Apply OneHotEncoder(handle_unknown='ignore') to categorical features
column_transformer = ColumnTransformer(
    transformers=[
        ('num',StandardScaler(),num_cols),
        ('cat',OneHotEncoder(handle_unknown='ignore'),cat_cols)
    ]
)

# TODO: Create XGBClassifier with:
#       objective='binary:logistic'
#       eval_metric='logloss'
#       random_state=42
clf = xgb.XGBClassifier(objective='binary:logistic',eval_metric='logloss',random_state=42)
# TODO: Create a Pipeline combining preprocessor + model
pipeline = Pipeline(steps=[
    ('preprocessor',column_transformer),
    ('model',clf)
])
# TODO: Train thepip model
pipeline.fit(X_train,y_train)
# TODO: Predict on test set
y_pred = pipeline.predict(X_test)
# TODO: Compute accuracy
acc = accuracy_score(y_test,y_pred)
# TODO: Print baseline accuracy
print(f"Baseline Accuracy:{acc:.4f}")
# TODO: Try combinations of:

n_estimators_list = [50, 150, 300]
max_depth_list = [3, 5, 7]
learning_rate_list = [0.01, 0.1, 0.2]

# TODO:
# Loop over all combinations
# Train model
# Compute test accuracy
# Track:
#   best_accuracy
#   worst_accuracy
best = 0.0
worst = 1.0
best_params = None
for i in n_estimators_list:
    for j in max_depth_list:
        for k in learning_rate_list:
            params = {
                'n_estimators':i,
                'max_depth':j,
                'learning_rate':k
            }
            model = xgb.XGBClassifier(**params)
            pipeline = Pipeline(steps=[
                ('preprocessor',column_transformer),
                ('model',model)
            ])
            pipeline.fit(X_train,y_train)
            y_pred = pipeline.predict(X_test)
            acc = accuracy_score(y_test,y_pred)
            if acc < worst:
                worst = acc
            if acc >= best:
                best = acc   
                best_params = params 
# TODO:
# Compute training accuracy
# Compare with test accuracy
            # y_train_pred = pipeline.predict(X_train)
            # train_acc = accuracy_score(y_train,y_train_pred)
            # print(f"Train accuracy:{train_acc:.4f}")
            # print(f"Test accuracy:{acc:.4f}") 
# TODO:
# Extract trained XGBoost model
extracted_model = xgb.XGBClassifier(**best_params)
best_pipeline = Pipeline(steps=[
    ('preprocessor',column_transformer),
    ('model',extracted_model)
])
best_pipeline.fit(X_train,y_train)
xgb_model = best_pipeline.named_steps['model']
# TODO:
# Plot feature importance (importance_type='gain')
import matplotlib.pyplot as plt
plot_importance(xgb_model,importance_type='gain')
plt.show()
# TODO:
# Save model as JSON
xgb_model.save_model('best_xgb.json')
# TODO:
# Load model back
loaded = xgb.XGBClassifier()
loaded.load_model('best_xgb.json')
# TODO:
# Verify loaded model accuracy
X_test = column_transformer.transform(X_test)
y_load_pred = loaded.predict(X_test)
load_acc = accuracy_score(y_test,y_load_pred)
print(f"Loaded_model_acc:{load_acc:.4f}")

