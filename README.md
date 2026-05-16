# MachineLearning

This is my project for the **Artificial Intelligence** course at the Faculty of Automatic Control and Computer Science, Politehnica University of Bucharest (Series CC, Year 3).

The goal was to apply a full ML pipeline on the **Job Salary Prediction Dataset** — from data exploration all the way to model evaluation.

## What this project does

There are two prediction tasks:
- **Classification** — predict the type of vacation an employee gets (`No Vacation`, `Small`, `Medium`, `Large`)
- **Regression** — predict the employee's `salary`

Tasks:

### 1. Exploratory Data Analysis (EDA)
- Analyzed numerical attributes using boxplots and summary statistics (mean, std, percentiles)
- Analyzed categorical attributes using histograms and value counts
- Checked for missing values and class imbalance
- Correlation analysis: Pearson for numerical pairs, Chi-Square for categorical pairs
- Found that `experience_years` and `total_days_worked` are perfectly correlated (r=1.00), so I dropped one of them

### 2. Preprocessing
- **Missing values**: `remote_work` had ~30% missing — imputed with mode ("Hybrid"); numerical columns imputed with median
- **Outliers**: detected and replaced using IQR method (mainly `skills_count` and `aggregated_score`)
- **Encoding**: Ordinal Encoding for `education_level`, `company_size`, `skill_bracket`; One-Hot Encoding for `job_title`, `industry`, `location`; Label Encoding for `remote_work`
- **Scaling**: StandardScaler applied to all numerical features

### 3. Classification
Started with a Decision Tree baseline and ran experiments from there:

| Model | Accuracy | F1 |
|---|---|---|
| DT baseline (depth=5) | 0.7139 | 0.7074 |
| DT (depth=10) | 0.7401 | 0.7382 |
| DT (depth=10, msl=50) | 0.7449 | 0.7407 |
| Random Forest (n=200, depth=10) | 0.7127 | 0.6859 |

Best model: **Decision Tree with depth=10 and min_samples_leaf=50**

### 4. Regression
Started with a Linear Regression baseline, then tried Ridge and Lasso regularization:

| Model | R² (val) | MAE (val) |
|---|---|---|
| Linear Regression | 0.9522 | 6289 |
| Ridge (alpha=1.0) | 0.9522 | 6289 |
| Lasso (alpha=10.0) | 0.9522 | 6289 |

All three models performed almost identically, which suggests the data has a very clear linear structure and regularization doesn't add much here.

## Libraries used

- `pandas`, `numpy`
- `scikit-learn`
- `matplotlib`, `seaborn`
