# Pandas Cheat Sheet – Kaggle Course Summary

## 1. Creating, Reading, and Writing

```python
pd.read_csv('file.csv')
df.to_csv('file.csv', index=False)
pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
```

## 2. Indexing, Selecting & Assigning

### `.iloc[]` — Integer-location based indexing:
```python
df.iloc[0]           # First row
df.iloc[:, 1]        # Second column
df.iloc[0:3, 1:3]    # First 3 rows, columns 1-2
```

### `.loc[]` — Label-based indexing:
```python
df.loc[3]                        # Row with label 3
df.loc[:, 'column']              # Column 'column'
df.loc[0:5, ['col1', 'col2']]    # Rows 0-5, specific columns
```

### Filtering and Assigning:
```python
df[df['col'] > 50]               # Filter rows where col > 50
df['total'] = df['A'] + df['B']  # Create new column
```

## 3. Summary Functions and Maps

```python
df.describe()        # Statistical summary
df.mean()            # Mean of each column
df.median()          # Median of each column
df.min()             # Minimum of each column
df.max()             # Maximum of each column
df['col'].value_counts()  # Count unique values

# Apply functions
df['col'].apply(lambda x: x ** 2)
df.apply(np.sum, axis=0)  # Sum of columns
df.apply(np.sum, axis=1)  # Sum of rows

# Map values
df['col'].map({'yes': 1, 'no': 0})
```

## 4. Grouping and Aggregating

### `groupby()`:
```python
df.groupby('category')['value'].mean()
df.groupby(['A', 'B']).size()
df.groupby('team')[['score', 'assists']].sum()
```

### `agg()`:
```python
df.groupby('category').agg({'price': ['mean', 'max']})
```

## 5. Sorting Data

### `sort_values()`:
```python
df.sort_values('col')
df.sort_values(by=['col1', 'col2'], ascending=[True, False])
```

### `sort_index()`:
```python
df.sort_index()       # Sort rows by index
df.sort_index(axis=1) # Sort columns
```

## 6. Data Types and Missing Data

```python
df.dtypes                # Get column data types
df['col'].astype('int')  # Convert column type
df.isnull()              # Boolean mask of null values
df.dropna()              # Drop rows with any null values
df.fillna(0)             # Replace nulls with 0
df.fillna(method='ffill') # Forward fill (propagate last valid value)
```

## 7. Renaming and Combining

```python
df.rename(columns={'old': 'new'})     # Rename columns
pd.concat([df1, df2])                 # Append dataframes
pd.merge(df1, df2, on='key', how='inner')  # Join dataframes
```