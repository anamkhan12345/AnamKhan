# Kaggle Data Visualization Cheat Sheet

## 📈 Trends
**Pattern of change over time**

### `sns.lineplot()`
- **Best for**: Showing trends over a period of time
- **Features**: Multiple lines can show trends in different groups
- **Use case**: Time series data, tracking changes over periods

```python
sns.lineplot(data=df, x='date', y='value')
sns.lineplot(data=df, x='date', y='value', hue='category')  # Multiple groups
```

---

## 🔗 Relationships
**Understanding connections between variables**

### `sns.barplot()`
- **Best for**: Comparing quantities across different groups
- **Use case**: Categorical comparisons, group performance metrics

```python
sns.barplot(data=df, x='category', y='value')
```

### `sns.heatmap()`
- **Best for**: Finding color-coded patterns in tables of numbers
- **Use case**: Correlation matrices, pivot tables, confusion matrices

```python
sns.heatmap(data=correlation_matrix, annot=True, cmap='coolwarm')
```

### `sns.scatterplot()`
- **Best for**: Relationship between two continuous variables
- **Features**: Color-coding can show relationship with third categorical variable
- **Use case**: Finding correlations, identifying outliers

```python
sns.scatterplot(data=df, x='variable1', y='variable2')
sns.scatterplot(data=df, x='variable1', y='variable2', hue='category')  # Third variable
```

### `sns.regplot()`
- **Best for**: Linear relationships with regression line
- **Features**: Makes linear relationships easier to identify
- **Use case**: Predictive analysis, trend confirmation

```python
sns.regplot(data=df, x='variable1', y='variable2')
```

### `sns.lmplot()`
- **Best for**: Multiple regression lines for different groups
- **Features**: Useful when scatter plot contains multiple color-coded groups
- **Use case**: Comparing trends across categories

```python
sns.lmplot(data=df, x='variable1', y='variable2', hue='category')
```

### `sns.swarmplot()`
- **Best for**: Relationship between continuous and categorical variables
- **Features**: Shows distribution of continuous variable within each category
- **Use case**: Comparing distributions across groups

```python
sns.swarmplot(data=df, x='category', y='continuous_variable')
```

---

## 📊 Distribution
**Showing possible values and their likelihood**

### `sns.histplot()`
- **Best for**: Distribution of a single numerical variable
- **Features**: Shows frequency of value ranges
- **Use case**: Understanding data spread, identifying skewness

```python
sns.histplot(data=df, x='variable')
sns.histplot(data=df, x='variable', bins=30)  # Custom bin count
```

### `sns.kdeplot()`
- **Best for**: Smooth distribution estimation
- **Features**: Can show single variable or 2D distributions
- **Use case**: Smooth density curves, bivariate distributions

```python
sns.kdeplot(data=df, x='variable')  # 1D KDE
sns.kdeplot(data=df, x='var1', y='var2')  # 2D KDE
```

### `sns.jointplot()`
- **Best for**: Simultaneous 2D KDE with individual variable KDE plots
- **Features**: Combines scatter plot with marginal distributions
- **Use case**: Comprehensive bivariate analysis

```python
sns.jointplot(data=df, x='variable1', y='variable2', kind='kde')
sns.jointplot(data=df, x='variable1', y='variable2', kind='scatter')
```

---

## 🎯 Quick Reference

| **Purpose** | **Chart Type** | **Variables** | **Best Use Case** |
|-------------|----------------|---------------|-------------------|
| **Trends** | Line Plot | Time + Numeric | Time series analysis |
| **Compare Groups** | Bar Plot | Categorical + Numeric | Group comparisons |
| **Patterns in Tables** | Heatmap | Matrix data | Correlations, pivot tables |
| **Two Continuous** | Scatter Plot | Numeric + Numeric | Find relationships |
| **Linear Trends** | Regression Plot | Numeric + Numeric | Predictive analysis |
| **Multiple Groups** | LM Plot | Numeric + Numeric + Category | Group trend comparison |
| **Categorical Scatter** | Swarm Plot | Category + Numeric | Distribution by group |
| **Single Distribution** | Histogram | Numeric | Data spread analysis |
| **Smooth Distribution** | KDE Plot | Numeric (1D/2D) | Density estimation |
| **Bivariate + Marginals** | Joint Plot | Numeric + Numeric | Comprehensive analysis |

---

## 💡 Pro Tips

- **Choose the right chart**: Match your data type and question to the appropriate visualization
- **Color coding**: Use `hue` parameter to add categorical dimensions
- **Multiple variables**: Combine plots to show complex relationships
- **Distribution first**: Always understand your data distribution before modeling