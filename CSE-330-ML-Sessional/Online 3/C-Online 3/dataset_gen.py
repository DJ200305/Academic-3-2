import numpy as np
import pandas as pd

np.random.seed(42)

n = 1000

df = pd.DataFrame({
    "age": np.random.randint(18, 65, n),
    "salary": np.random.randint(20000, 120000, n),
    "account_balance": np.random.randint(1000, 50000, n),
    "gender": np.random.choice(["Male", "Female"], n),
    "city": np.random.choice(["Dhaka", "Chittagong", "Khulna", "Rajshahi"], n),
    "membership_type": np.random.choice(["Basic", "Silver", "Gold"], n),
})

# Create target (purchased)
df["purchased"] = (
    (df["salary"] > 60000).astype(int)
    + (df["membership_type"] == "Gold").astype(int)
    + (df["account_balance"] > 20000).astype(int)
)

df["purchased"] = (df["purchased"] > 1).astype(int)

df.to_csv("customer_data.csv", index=False)

print("Dataset created successfully.")