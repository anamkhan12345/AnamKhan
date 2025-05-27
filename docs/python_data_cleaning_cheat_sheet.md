
# 🧼 Kaggle Data Cleaning Cheat Sheet (Pandas)

---

## 📋 1. Handling Missing Values

### ➕ Check for missing values
```python
df.isnull().sum()
```

### ➖ Drop missing values
```python
df.dropna()              # Drop rows with any missing values
df.dropna(axis=1)        # Drop columns with missing values
```

### ✏️ Fill missing values
```python
df.fillna(0)                              # Fill with constant
df['col'].fillna(df['col'].mean())       # Fill with column mean
```

### 🎯 Replace specific values
```python
df.replace("?", np.nan)
```

---

## 🧼 2. Handling Inconsistent Data

### 🔍 View unique values
```python
df['col'].value_counts(dropna=False)
```

### 🔁 Standardize text
```python
df['col'] = df['col'].str.lower()
df['col'] = df['col'].str.strip()
```

### 🔄 Replace wrong entries
```python
df['col'] = df['col'].replace({'mistke': 'mistake'})
```

---

## 📏 3. Data Type Conversions

### 🔍 Check types
```python
df.dtypes
```

### 🔁 Convert types
```python
df['col'] = df['col'].astype('int')
pd.to_numeric(df['col'], errors='coerce')  # Invalid entries become NaN
```

### ⏰ Convert to datetime
```python
df['date_col'] = pd.to_datetime(df['date_col'], errors='coerce')
```

---

## 🧪 4. Detecting and Fixing Outliers

### 📊 Summary statistics
```python
df.describe()
```

### 🧱 Filter by logical conditions
```python
df[df['col'] < 100]
```

---

## 🧹 5. String Methods

```python
df['col'].str.contains('abc')
df['col'].str.replace('$', '', regex=False)
df['col'].str.extract('(\d+)')
```

---

## ⚙️ 6. Advanced Cleaning Tips

### 📌 Chain multiple steps
```python
df['col'] = df['col'].str.lower().str.strip().replace({'unknwn': np.nan})
```

### 🧹 Clean column names
```python
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
```

---

## ✅ 7. Final Checks

```python
df.info()                 # Column types and non-null counts
df.isnull().sum()         # Missing values per column
df.duplicated().sum()     # Number of duplicate rows
df.drop_duplicates()      # Remove duplicate rows
```
