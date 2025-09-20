### 1. **MSE (Mean Squared Error)**

```python
mse = mean_squared_error(y_test, y_pred)
```

* **Definition:** Average of the squares of the errors (difference between predicted and actual values)
* **Formula:**

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

* **Interpretation:**

  * Lower MSE → predictions are closer to actual values
  * Squaring penalizes **large errors more**

---

### 2. **RMSE (Root Mean Squared Error)**

```python
rmse = mean_squared_error(y_test, y_pred, squared=False)  # or take sqrt of MSE
```

* **Definition:** Square root of MSE
* **Formula:**

$$
\text{RMSE} = \sqrt{\text{MSE}}
$$

* **Interpretation:**

  * Same as MSE, but in **same units as the target variable**
  * Easier to interpret than MSE

---

### 3. **R² Score (Coefficient of Determination)**

```python
r2 = r2_score(y_test, y_pred)
```

* **Definition:** Measures how much of the variance in `y` your model explains
* **Formula:**

$$
R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}
$$

* **Interpretation:**

  * R² = 1 → perfect prediction
  * R² = 0 → model predicts no better than the mean of `y`
  * R² < 0 → model is worse than just predicting the mean

---

### TL;DR

| Metric | What it measures   | Lower/Better?    | Notes                            |
| ------ | ------------------ | ---------------- | -------------------------------- |
| MSE    | Avg squared error  | Lower is better  | Penalizes large errors           |
| RMSE   | Square root of MSE | Lower is better  | Same units as target             |
| R²     | Explained variance | Higher is better | 1 = perfect, 0 = mean prediction |

---

⚡ Tip: Use **RMSE** if you want error in the same units as your target, and **R²** to quickly see model quality.
