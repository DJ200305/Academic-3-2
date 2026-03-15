import pandas as pd
import numpy as np

np.random.seed(42)

# 1. Minimum Data Imputation
def minimum_data_imputation(X):
    # TODO: Minimum data imputation
    
    return X_imputed # with NaNs imputed # of shape as X

# 2. Softmax
def softmax(z):
    # TODO: Softmax

    return value # of shape as z

# 3. Softmax + CE gradient
def softmax_ce_gradient(p, y):
    # TODO: Softmax + CE gradient

    return grad # of shape as p

# 4. Multi-Class Cross-Entropy Loss
def multiclass_ce_loss(p, y):
    # TODO: Multi-Class Cross-Entropy Loss

    return loss # only a scalar value

# 5. Load test data and evaluate accuracy
def inference(df_test, W, b):
    # TODO: 5. Load test data and evaluate accuracy

    return test_accuracy # test_accuracy


# ============================
# MAIN: MINIBATCH TRAINING + ACCURACY
# ============================
if __name__ == "__main__":
    df = pd.read_csv("train_data.csv", header=None)
    print("Data size:", df.shape)

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values.reshape(-1, 1)
    # One-hot encoding for y
    n_classes = len(np.unique(y))
    y_onehot = np.zeros((y.shape[0], n_classes))
    for i in range(y.shape[0]):
        y_onehot[i, y[i, 0]] = 1

    # Use minimum_data_imputation for any remaining NaN values
    print("NaN values before imputation:", np.isnan(X).sum())
    print("Applying minimum data imputation...")
    X = minimum_data_imputation(X)
    print("NaN values after imputation:", np.isnan(X).sum())

    n_samples, n_features = X.shape

    # Initialize parameters
    W = np.zeros((n_features, 4))
    b = np.zeros((1, 4))

    batch_size = 50
    learning_rate = 0.1
    num_epochs = 20

    print(f"\nTraining for {num_epochs} epochs with learning rate {learning_rate}...\n")

    # Training loop
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        # Shuffle data at the beginning of each epoch
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y_onehot[indices]
        
        for i in range(0, n_samples, batch_size):
            Xb = X_shuffled[i:i+batch_size]
            yb_onehot = y_shuffled[i:i+batch_size]

            outputs = Xb @ W + b
            preds = softmax(outputs)
            
            # Compute loss
            batch_loss = multiclass_ce_loss(preds, yb_onehot)
            epoch_loss += batch_loss
            num_batches += 1
            
            # Gradient of loss w.r.t predictions
            dpreds_doutputs = softmax_ce_gradient(preds, yb_onehot)
            
            # Gradients w.r.t W and b
            dW = Xb.T @ dpreds_doutputs
            db = np.sum(dpreds_doutputs, axis=0, keepdims=True)
            
            # Update weights
            W -= learning_rate * dW
            b -= learning_rate * db
        
        avg_loss = epoch_loss / num_batches
        
        # Calculate accuracy on full dataset
        outputs_all = X @ W + b
        y_pred = softmax(outputs_all)
        y_pred_classes = np.argmax(y_pred, axis=1).reshape(-1, 1)

        accuracy = np.mean(y_pred_classes == y) * 100
        
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.6f}, Accuracy: {accuracy:.2f}%")

    print("\n" + "="*50)
    print("Training completed!")
    print("="*50)

    # Final evaluation
    outputs_final = X @ W + b
    preds_final = softmax(outputs_final)
    final_loss = multiclass_ce_loss(preds_final, y_onehot)
    y_pred_classes_final = np.argmax(preds_final, axis=1).reshape(-1, 1)

    print(f"\nFinal Loss: {final_loss:.6f}")
    final_accuracy = np.mean(y_pred_classes_final == y) * 100
    print(f"Final Accuracy: {final_accuracy:.2f}%")

    print("\nTrained weights (W):", W.ravel())
    print(f"Trained bias (b): {b.ravel()}")

    # ============================
    # INFERENCE ON TEST DATA
    # ============================
    df_test = pd.read_csv("test_data.csv", header=None)
    
    t_accuracy = inference(df_test, W, b)

