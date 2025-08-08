import pandas as pd

# Replace with your actual file name if different
df = pd.read_csv("data.csv")
df.head()

print(df.info())
print(df['target'].value_counts())  # Replace 'target' with actual target column name
