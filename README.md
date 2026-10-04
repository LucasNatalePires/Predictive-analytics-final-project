# Predictive-analytics-final-project

## Objective
Final Continuous Assessment for the Predictive Data Analytics summer course (CCT College Dublin), applying four core predictive analytics techniques — **regression**, **classification**, **PCA + clustering**, and **time series forecasting** — each on a different dataset, with a dedicated EDA, modelling and evaluation stage per task.

---

## Task 1 — Regression: Cereal Ratings

**Objective:** predict a cereal's consumer rating from its nutritional profile.

**Dataset:** `Cereals_Rating.csv` — 76 cereals, nutritional attributes (calories, sugar, fiber, protein, sodium, etc.) and a continuous `rating` target. No nulls or duplicates found.

**EDA highlights:** calories and sugar show a strong **negative** correlation with rating; fiber/protein show a weaker, less clearly defined relationship.

**Encoding:** `mfr` and `type` converted via `pd.get_dummies`/`.map()` (dataset is small and static, so scikit-learn's `OneHotEncoder` wasn't necessary); `Name` dropped (no predictive value).

**Models & Results:**

| Model | RMSE | MSE | R² |
|---|---|---|---|
| **Linear Regression** | **1.7167** | **2.9472** | **0.9825** |
| Polynomial Regression | 2.1136 | 4.4671 | 0.9735 |
| Random Forest | 4.7651 | 22.7065 | 0.8653 |

**Conclusion:** Linear Regression performed best — the strong R² (98.25%) suggests rating is likely built from a near-linear formula weighting each nutrient. **Limitation:** only 76 records, a small sample for drawing strong conclusions.

---

## Task 2 — Classification: Online News Popularity

**Objective:** classify news articles by popularity level, based on engagement-related features.

**Dataset:** `OnlineNewsPopularity.csv` — 61 attributes (58 predictive, 2 non-predictive, 1 continuous `shares` metric). No suitable categorical target existed, so one was engineered.

**Target creation:** `shares` was split into 3 classes using quartiles — Not Popular (≤ 946), Popular (946–3,080), Very Popular (> 3,080, i.e. Q3 + 10%) — to separate genuinely viral articles from merely popular ones.

**Preparation:** column names stripped of stray whitespace; strong right-skew found across many features, addressed with `RobustScaler`; `stratify` used in the train/test split to manage class imbalance (Popular = 20,788 vs. Not Popular = 9,930).

**Models (via `GridSearchCV`) & Results:**

| Model | Accuracy | F1-score |
|---|---|---|
| **Random Forest** | **0.55** | **0.51** |
| KNN | 0.50 | 0.47 |
| Logistic Regression | 0.38 | 0.38 |

**Results & Limitations:** Random Forest's clear win over Logistic Regression indicates the relationship isn't linear. The confusion matrix showed heavy overlap between "Popular" and "Very Popular" (~50% confused) — the engineered classes capture `shares` well, but the other features don't differ enough between those two groups to separate them reliably. A looser "Very Popular" threshold (Q3 +20% or higher) could sharpen the distinction, at the cost of worsening class imbalance further.

---

## Task 3 — PCA & Clustering: Dry Bean Dataset

**Objective:** reduce dimensionality and group dry beans into their 7 real varieties using unsupervised learning, and test whether PCA improves clustering quality.

**Dataset:** `Dry_Bean_Dataset.csv` — 16 geometric features (area, perimeter, axis lengths, eccentricity, etc.) describing 7 bean types. No nulls; duplicates found and removed. Strong multicollinearity identified via the correlation heatmap.

**Scaling & PCA:** both `RobustScaler` and `StandardScaler` were tested (outlier-robust vs. standard approach). Under both, **4 principal components retained 95% of the variance** from the original 16 features.

**Choosing K:** the Elbow Method suggested K=5, the Silhouette Score suggested K=3 — but **K=7 was used**, matching the known number of real bean types, and validated after the fact with the Adjusted Rand Index.

**Results — With vs. Without PCA:**

| Metric | With PCA (4 components) | Without PCA (16 features) |
|---|---|---|
| Silhouette Score | **0.3360** | 0.3152 |
| Adjusted Rand Index (ARI) | **0.6807** | 0.6773 |

**Conclusion:** `RobustScaler` outperformed `StandardScaler` slightly (Silhouette 0.3360 vs 0.3271). PCA gave a modest but real improvement — most visible in the Silhouette Score, since reducing 16 features to 4 makes distances easier for K-Means to compute; the ARI barely moved, since all the class-separating information was already present before PCA (it just organised it more compactly). **Limitation:** perfect separation between bean types wasn't achieved — several varieties share overlapping geometric characteristics.

---

## Task 4 — Time Series Forecasting: Appliance Energy Use

**Objective:** forecast household appliance energy consumption (Wh), recorded every 10 minutes.

**Dataset:** `energydata.csv` — 29 attributes (28 features including temperature/humidity sensors across rooms, plus `date`), target = `Appliances` (Wh). No nulls or duplicates.

**Stationarity:** confirmed via the ADF test (p-value = 0) — no differencing required. ACF/PACF plots pointed to autoregressive behaviour.

**ARIMA — model comparison:**

| Metric | ARIMA(1,0,0) | ARIMA(2,0,0) |
|---|---|---|
| AIC | 178,394.95 | **178,302.94** |
| BIC | 178,417.95 | **178,333.61** |
| Ljung-Box Prob(Q) | 0.00 | **0.19** |

ARIMA(2,0,0) selected. **Error metrics:** MAE = 52.53 Wh, RMSE = 90.66, MAPE = 59.88% — a high relative error, given the target's median is only ~60 Wh.

**SARIMA:** daily seasonality (s=144 at 10-min resolution) was too memory-intensive, so data was aggregated to hourly (s=24) — reducing the sample from 15,788 to 2,632 points, meaning AIC/BIC aren't directly comparable between the two models. **Error metrics:** MAE = 54.46, RMSE = 96.34, MAPE = 40.17%. Seasonal coefficient `ar.S.L24 = 0.9981` (very strong daily pattern); Ljung-Box Prob(Q) = 0.71 (residuals well captured).

**Final comparison:** SARIMA's MAE/RMSE were marginally higher than ARIMA's, but its much better Ljung-Box result (0.71 vs 0.19) and far lower MAPE (40.17% vs 59.88%) indicate it captures the series' real daily pattern better — **SARIMA is the more suitable model**, even though the two aren't perfectly comparable due to the different data resolutions used.

**Limitations:** both models struggled to capture extreme consumption peaks (confirmed via Jarque-Bera and Q-Q plots); SARIMA was only tested with one seasonal configuration due to computational cost; the dataset's own temperature/humidity variables were not used as exogenous predictors — a natural next step.

---

## Overall Conclusion
Across the four tasks, the strongest results came where the underlying relationship was closest to linear or had a clear structural driver (Task 1's cereal ratings, R² = 0.98) or a well-defined seasonal pattern (Task 4's SARIMA). Weaker results (Task 2's 55% accuracy) came from an engineered, inherently noisy target where the available features didn't fully separate the classes — a useful reminder that model choice can't fully compensate for a target variable with a weak signal-to-noise ratio in the first place.

## Files in this repository

| File | Description |
|---|---|
| [`CA_LucasPires.ipynb`](./notebook/CA_LucasPires.ipynb) | Full notebook — all 4 tasks, EDA, modelling and evaluation |
| [`Cereals_Rating.csv`](./data/Cereals_Rating.csv) | Dataset for Task 1 (regression) |
| [`OnlineNewsPopularity.csv`](./data/OnlineNewsPopularity.csv) | Dataset for Task 2 (classification) |
| [`Dry_Bean_Dataset.csv`](./data/Dry_Bean_Dataset.csv) | Dataset for Task 3 (PCA + clustering) |
| [`energydata.csv`](./data/energydata.csv) | Dataset for Task 4 (time series forecasting) |
| [`PDA_2026_-_Final_Project.docx`](./docs/PDA%202026%20-%20Final%20Project__.docx) | Assignment brief — CCT College Dublin |
