import pandas as pd

# Load dataset
df = pd.read_csv("../datasets/titanic.csv")

# Basic inspection
print("First 5 rows:\n")
print(df.head(10))

print("\nShape:", df.shape)

print("\nColumns:\n", df.columns)

print("\nMissing values:\n", df.isnull().sum())

print("\nData types:\n", df.dtypes)
print("row*cols",df.size)