import pandas as pd
import numpy as np

np.random.seed(42)

# 1. Min-Max Scaling
def min_max_scaling(x):
    # TODO: Implement (x - min) / (max - min) [cite: 6]
    # Hint: Use axis=0 to calculate min/max per feature
    X_scaled = (x-np.min(x))/(np.max(x)-np.min(x))
    return X_scaled

# 2. Sigmoid Activation
def sigmoid(x):
    # TODO: Implement 1 / (1 + exp(-x)) [cite: 8, 9]
    return 1/(1+np.exp(-x))

# 3. Sigmoid Gradient
def sigmoid_gradient(x, dout=1):
    # TODO: Implement dL/dx = dout * sigma(x) * (1 - sigma(x)) [cite: 10, 11, 13]
    return dout*sigmoid(x)*(1-sigmoid(x))

# 4. Binary Cross-Entropy (BCE) Loss
def bce_loss(y_pred, y_true):
    # TODO: Implement -1/n * sum(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
    # Note: Ensure it returns a scalar value [cite: 14]
    return (-1)*np.mean(y_true*np.log(y_pred)+(1-y_true)*np.log(1-y_pred))

# 5. BCE Loss Gradient
def bce_loss_gradient(y_pred, y_true):
    # TODO: Implement gradient of BCE w.r.t y_pred:
    # grad = -1/n * ( (y_true/y_pred) - (1-y_true)/(1-y_pred) ) [cite: 15, 16]
    return (-1/y_true.shape[0])*((y_true/y_pred)-((1-y_true)/(1-y_pred)))

# ============================
# MAIN: MINIBATCH TRAINING
# ============================
if __name__ == "__main__":
    # Generate dummy data: 200 samples, 3 features
    X = np.random.rand(200, 3)
    y = np.random.randint(0, 2, (200, 1))

    # Normalize
    X = min_max_scaling(X)

    n_samples, n_features = X.shape
    W = np.zeros((n_features, 1))
    b = 0.0

    batch_size = 20
    learning_rate = 0.01
    num_epochs = 5

    for epoch in range(num_epochs):
        indices = np.random.permutation(n_samples)
        X_sh = X[indices]
        y_sh = y[indices]
        
        for i in range(0, n_samples, batch_size):
            Xb = X_sh[i:i+batch_size]
            yb = y_sh[i:i+batch_size]
            
            # Forward Pass
            logits = Xb @ W + b
            preds = sigmoid(logits)
            
            # Backward Pass
            dloss_dpreds = bce_loss_gradient(preds, yb)
            dpreds_dlogits = sigmoid_gradient(logits, dout=dloss_dpreds)
            
            dW = Xb.T @ dpreds_dlogits
            db = np.sum(dpreds_dlogits)
            
            # Update
            W -= learning_rate * dW
            b -= learning_rate * db
            
        print(f"Epoch {epoch+1} complete.")