# Importing required libraries
import numpy as np
import pandas as pd
from apyori import apriori

# Step 1: Load the dataset
dataset = pd.read_csv("Market_Basket_Optimisation.csv", header=None)

# Step 2: Convert dataset into a list of transactions
transactions = []
for i in range(len(dataset)):
    transactions.append(
        [
            str(dataset.values[i, j])
            for j in range(len(dataset.columns))
            if str(dataset.values[i, j]) != "nan"
        ]
    )

# Step 3: Apply Apriori Algorithm
rules = apriori(
    transactions, min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2
)

# Convert rules into a list
results = list(rules)


# Step 4: Organizing Results into a DataFrame
def inspect(results):
    lhs = [tuple(result[2][0][0])[0] for result in results]
    rhs = [tuple(result[2][0][1])[0] for result in results]
    support = [result[1] for result in results]
    confidence = [result[2][0][2] for result in results]
    lift = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, support, confidence, lift))


# Creating a DataFrame for better readability
output_df = pd.DataFrame(
    inspect(results),
    columns=["Left_Hand_Side", "Right_Hand_Side", "Support", "Confidence", "Lift"],
)

# Displaying the results
print(output_df)
