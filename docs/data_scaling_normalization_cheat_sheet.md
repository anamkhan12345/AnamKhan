
# ⚖️ Data Scaling & Normalization Cheat Sheet (Python / Pandas / Scikit-learn)

---

## 📌 Why Scale or Normalize?
- Features have different units or ranges (e.g., age vs. income).
- Algorithms like k-NN, SVM, logistic regression, neural nets benefit from scaling.
- Improves model performance and convergence speed.

---

## 🔧 Common Methods

---

### ✅ 1. Standardization (Z-score Normalization)
Centers data around 0 with unit variance.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_array = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_array, columns=df.columns, index=df.index)
```

\`\`\`math
z = \frac{x - \mu}{\sigma}
\`\`\`

---

### ✅ 2. Min-Max Scaling
Scales features to [0, 1] range.

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaled_array = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_array, columns=df.columns, index=df.index)
```

---

### ✅ 3. Robust Scaling
Uses median and IQR — useful when data has outliers.

```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
scaled_array = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_array, columns=df.columns, index=df.index)
```

---

### ✅ 4. Normalize Each Sample (L2 norm)
Scales each row (sample) to unit norm.

```python
from sklearn.preprocessing import Normalizer

scaler = Normalizer()
normalized_array = scaler.fit_transform(df)
normalized_df = pd.DataFrame(normalized_array, columns=df.columns, index=df.index)
```

---

## 🧠 Tips

### 🔍 Select only numerical features
```python
numeric_df = df.select_dtypes(include='number')
```

### 🧼 Preserve index and column names
```python
pd.DataFrame(scaled_array, columns=df.columns, index=df.index)
```

### 📊 When to Use Each
| Method         | Good For                         | Sensitive to Outliers |
|----------------|----------------------------------|------------------------|
| StandardScaler | Most ML algorithms               | Yes                    |
| MinMaxScaler   | Neural nets, image data          | Yes                    |
| RobustScaler   | Heavy-tailed / skewed distributions | No                  |
| Normalizer     | Cosine similarity / text data    | N/A (row-wise scaling) |

---

