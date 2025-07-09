# 📘 Day 2: NumPy & Pandas Basics

## 🧮 NumPy – Numerical Python

### ✅ Key Concepts
- **Array Creation**: `np.array()`, `np.zeros()`, `np.ones()`, `np.arange()`, `np.linspace()`
- **Array Reshaping**: `.reshape()`, `.flatten()`
- **Slicing & Indexing**: `arr[1:5]`, `arr[:, 2]`
- **Broadcasting**: Operations between arrays of different shapes
- **Mathematical Operations**: `np.mean()`, `np.std()`, `np.sum()`, `np.dot()`, `np.exp()`
- **Random Generation**:
  - `np.random.rand()`, `np.random.randn()`, `np.random.randint()`
  - `np.random.seed()` for reproducibility

### 💡 Mini Challenge
- Created a `10x10` matrix
  - Replaced diagonals with `1`
  - Normalized matrix between 0 and 1
  - Extracted prime numbers

---

## 🐼 Pandas – Python Data Analysis Library

### ✅ Core Concepts
- **Data Structures**: `Series`, `DataFrame`
- **Data Loading**: `pd.read_csv()`, `pd.read_excel()`
- **Data Exploration**:
  - `.head()`, `.info()`, `.describe()`
  - `.shape`, `.columns`, `.dtypes`

### 📊 Data Manipulation
- **Filtering & Indexing**: `df[df['Age'] > 25]`
- **Sorting**: `df.sort_values(by='Fare', ascending=False)`
- **Grouping**: `df.groupby('Sex')['Survived'].mean()`
- **Aggregation**: `sum()`, `mean()`, `count()`, `agg()`
- **Handling Nulls**: `df.dropna()`, `df.fillna()`
- **Merging & Joining**: `pd.concat()`, `pd.merge()`

### 💡 Titanic Dataset Practice
- Loaded the Titanic dataset from URL
- Cleaned null values from `Age`, `Embarked`
- Grouped by `Sex` → Calculated survival rate
- Extracted top 3 passengers by `Fare`
- Normalized `Age` column using Min-Max Scaling

---

## 🧰 Tools Used
- **Libraries**: `numpy`, `pandas`
- **Platform**: Jupyter Notebook
- **Dataset**: Titanic.csv from GitHub

---

## 🔚 Summary
Today I mastered the foundational tools for AI/ML:
- `NumPy` for numerical computing & matrix handling
- `Pandas` for dataset cleaning, transformation, and analysis

These will be my go-to tools for all ML preprocessing and EDA tasks going forward.

---
