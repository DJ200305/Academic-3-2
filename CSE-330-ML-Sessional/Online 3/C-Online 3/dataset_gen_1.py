import numpy as np
import pandas as pd

np.random.seed(42)
n = 2000

age = np.random.randint(18, 65, n)
salary = np.random.randint(20000, 150000, n)
balance = np.random.randint(1000, 80000, n)
transactions = np.random.randint(1, 200, n)

gender = np.random.choice(["Male", "Female"], n)
city = np.random.choice(["Dhaka", "Chittagong", "Khulna", "Rajshahi", "Sylhet"], n)
membership = np.random.choice(["Basic", "Silver", "Gold", "Platinum"], n)

# Non-linear + interaction-based target
score = (
    (salary > 80000).astype(int)
    + (balance > 30000).astype(int)
    + (transactions > 100).astype(int)
    + (membership == "Gold").astype(int)
    + (membership == "Platinum").astype(int)
)

# Add interaction effect
score += ((age < 30) & (transactions > 120)).astype(int)

target = (score >= 3).astype(int)

df = pd.DataFrame({
    "age": age,
    "salary": salary,
    "account_balance": balance,
    "transactions": transactions,
    "gender": gender,
    "city": city,
    "membership_type": membership,
    "target": target
})

df.to_csv("advanced_customer_data.csv", index=False)
print("advanced_customer_data.csv created successfully.")