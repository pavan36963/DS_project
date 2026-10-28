# %%
#Replace missing data with mean of data.
import pandas as pd
import numpy as np
df = pd.DataFrame({'Age': [25,30,np.nan, 40, 35], 'Department': ['HR', 'Finance', 'Finance', np.nan, 'IT']})
print(df)
df['Age']=df['Age'].fillna(df['Age'].mean())
df['Department']=df['Department'].fillna(df['Department'].mode()[0])
print(df)
# %%
#Forward fill -> replace missing values with prev values
import pandas as pd
import numpy as np
df = pd.DataFrame({'Age': [25, 30, np.nan, 40, 35], 'Department': ['HR', 'Finance', 'Finance', np.nan, 'IT']})
print("Original Dataset (with missing values): ")
print(df)
df_ffill = df.copy()
df_ffill.ffill(inplace=True)
print(df_ffill)

# %%
#Backward fill -> replace missing values with next values
import pandas as pd
import numpy as np
df = pd.DataFrame({'Age': [25, 30, np.nan, 40, 35], 'Department': ['HR', 'Finance', 'Finance', np.nan, 'IT']})
print("Original Dataset (with missing values): ")
print(df)
df_bfill = df.copy()
df_bfill.bfill(inplace=True)
print(df_bfill)
# %%
#Drop rows with missing values
import pandas as pd
import numpy as np

df = pd.DataFrame({
    "Age": [25, 30, np.nan,40,35],
    "Department": ["HR", "Finance", np.nan, "Finance", "IT"]
})
print("Original Dataset")
print(df)

df_drop_rows = df.dropna()
print("After removing rows:\n", df_drop_rows)
# %%
#Drop cols with missing values
df_drop_cols = df.dropna(axis=1)
print("After dropping columns:\n", df_drop_cols)
# %%
df_drop_r = df.dropna(axis=0)
print("After dropping rows:\n", df_drop_r)
# %%
#Removing duplicate values in dataset
import pandas as pd
df = pd.DataFrame({
    'ID': [1,2,2,3,4,4],
    'Name': ['Alice', 'Bob', 'Bob', 'Charlie', 'David', 'David'],
    'Age': [25, 30, 30, 35, 40, 40]
})
print("Original Data:\n", df)
df_exact = df.drop_duplicates()
print("After Exact Match Removal:\n", df_exact)
# %%
#Subset Based Removal
#Removal of duplicates based only on 'ID'
import pandas as pd
df = pd.DataFrame({
    'ID': [1,2,2,3,4,4],
    'Name': ['Alice', 'Bob', 'Bob', 'Ch\arlie', 'David', 'David'],
    'Age': [25, 30, 30, 35, 40, 40]
})
print("Original Data:\n", df)
#Remove based only on ID
df_subset_id = df.drop_duplicates(subset=['ID'])
print("\nAfter Subset-Based removal (ID):\n", df_subset_id)
#Remove based only on 'Name
df_subset_name = df.drop_duplicates(subset=['Name'])
print("\nAfter Subset-Based removal (Name):\n", df_subset_name)
# %%
#Correcting Inconsistent date Formats
import pandas as pd
#sample dataset with inconsistent date formats
df = pd.DataFrame({
    'Date': ['2025-01-05', '05/01/2025', 'Jan 5, 2025', '2025.01.05']
})
print("Original Data:\n", df)
#convert all to standard ISO format (year-month-date)
df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.strftime('%Y-%m-%d')
print(df)


# %%
#Case Normalization (lower/upper)
df = pd.DataFrame({
    'Name': ['Alice', 'BOB', 'charlie', 'DAVID']
})
#convert all to lowercase
df['Name_lower'] = df['Name'].str.lower()
#Convert all to uppercase
df['Name_upper'] = df['Name'].str.upper()
print(df)

# %%
