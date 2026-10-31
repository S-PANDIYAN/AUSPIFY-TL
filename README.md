# Netflix Machine Learning & Data Science Analytics Suite

An end-to-end portfolio of six machine learning, natural language processing, and predictive analytics projects built on the Netflix catalog dataset. Each project is implemented as a self-contained, interactive Jupyter Notebook (`.ipynb`) optimized for direct execution in **Google Colab** and **VS Code / JupyterLab**.

---

## Project Directory & Task Overview

| Task | Level | Notebook File | Objective | Primary Algorithms & Tools |
| :--- | :--- | :--- | :--- | :--- |
| **Task 1** | Easy | [`Netflix_Recommendation_System.ipynb`](./Netflix_Recommendation_System.ipynb) | Content-based movie & TV show recommender | TF-IDF Vectorization, Cosine Similarity, Jaccard Index |
| **Task 2** | Easy | [`Content_Type_Prediction.ipynb`](./Content_Type_Prediction.ipynb) | Predict whether a title is a Movie or TV Show | Binary Classification, ColumnTransformer, Logistic Regression, Random Forest, Gradient Boosting |
| **Task 3** | Medium | [`Audience_Rating_Classification.ipynb`](./Audience_Rating_Classification.ipynb) | Classify content into audience maturity tiers | Multi-Class Classification, Decision Trees, Random Forest, `GridSearchCV` 5-Fold Cross-Validation |
| **Task 4** | Medium | [`Content_Segmentation_Clustering.ipynb`](./Content_Segmentation_Clustering.ipynb) | Unsupervised grouping of catalog content | K-Means Clustering, Elbow Curve, Silhouette Analysis, PCA 2D Projection, Persona Profiling |
| **Task 5** | Advanced | [`Netflix_Trend_Forecasting.ipynb`](./Netflix_Trend_Forecasting.ipynb) | Time series release volume trend forecasting | Seasonal Decomposition, Autoregressive Lag Engineering, Holt-Winters Exponential Smoothing, Random Forest Regressor |
| **Task 6** | Advanced | [`Content_Success_Analytics_Engine.ipynb`](./Content_Success_Analytics_Engine.ipynb) | Enterprise content success scoring & automated BI | End-to-End MLOps Pipeline, Robust Feature Engineering, Automated AI Insights, Executive Pre-Release Simulator |

---

## Dataset Description

The analysis is driven by `Dataset.csv`, containing **8,790+** streaming records with 10 core attributes:

| Column | Data Type | Description | Role in ML Workflows |
| :--- | :--- | :--- | :--- |
| `show_id` | String | Unique catalog identifier (`s1`, `s2`, ...) | Primary key index |
| `type` | Categorical | Format of content (`Movie` or `TV Show`) | Target in Task 2; feature in Tasks 1, 3, 4, 6 |
| `title` | String | Title name | Query search key and catalog identifier |
| `director` | Text | Director(s) name(s) | Style/creator feature; catalog prominence scoring |
| `country` | Categorical | Country or countries of production | Geographic reach, regional preference & co-production analysis |
| `date_added` | Datetime | Date title was added to Netflix | Temporal base for Task 5 time series & Task 6 acquisition lag |
| `release_year` | Integer | Original global debut year | Vintage gap analysis & era tracking |
| `rating` | Categorical | Maturity rating (`TV-MA`, `PG-13`, `TV-PG`, etc.) | Target in Task 3; core audience filter across tasks |
| `duration` | String / Numeric | Runtime in minutes (Movies) or seasons (TV Shows) | Format length indicator & viewer retention proxy |
| `listed_in` | Text | Comma-separated genre categories | Semantic feature extraction via TF-IDF |

---

## Task-by-Task Implementation Summary

### Task 1: Content Recommendation System
* **Goal:** Recommend the top 5 most contextually relevant titles for any selected movie or TV show.
* **Workflow:**
  1. **Step 1: Feature Preparation:** Cleans missing/placeholder values and creates a consolidated "metadata soup" combining format, weighted genres, tokenized director names, country, and maturity ratings.
  2. **Step 2: Vectorization:** Converts metadata text into high-dimensional numerical vectors using `TfidfVectorizer` with sublinear frequency scaling.
  3. **Step 3: Similarity Matrix:** Calculates an $8,790 \times 8,790$ symmetric pairwise **Cosine Similarity** matrix.
  4. **Step 4: Recommendations:** Implements `get_recommendations(title, top_n=5)` with case-insensitive and partial title matching.
  5. **Step 5: Quality Evaluation:** Benchmarks recommendations with **Genre Jaccard Overlap Index (0.820/1.000)**, **Format Consistency (100%)**, and **Mean Confidence (79.7%)**.
  6. **Interactive Demo:** Colab form allows users to enter any title and view recommendations with match percentages.

---

### Task 2: Content Type Prediction Model
* **Goal:** Train supervised binary classification models to predict whether a title is a **Movie** or a **TV Show** using content metadata.
* **Workflow:**
  1. **Step 1: Feature Selection:** Identifies predictors (`listed_in`, `rating`, `country`, `release_year`, `has_director`). **Prevents target leakage** by deliberately excluding raw duration units (`"min"` vs `"Season"`).
  2. **Step 2: Categorical Encoding:** Applies `LabelEncoder` for target labels (`Movie` $\to 0$, `TV Show` $\to 1$) and `ColumnTransformer` (TF-IDF for genres, One-Hot Encoding for categories, `StandardScaler` for release year). Stratified 80/20 train-test split.
  3. **Step 3: Model Training:** Trains **Logistic Regression**, **Decision Tree**, **Random Forest**, and **Gradient Boosting**.
  4. **Step 4: Performance Evaluation:** Evaluates Confusion Matrices, Precision, Recall, F1-Scores, and ROC-AUC curves.
  5. **Step 5: Model Comparison:** Synthesizes a multi-metric benchmark table and bar chart, followed by an interactive predictor form.

---

### Task 3: Netflix Audience Rating Classification
* **Goal:** Build a multi-class classification model that maps content attributes to audience maturity levels.
* **Workflow:**
  1. **Step 1: Category Analysis:** Maps 10+ raw ratings into 4 standardized tiers:
     - `Adults (18+)`: `TV-MA`, `R`, `NC-17`, `UR`
     - `Teens (13-14+)`: `TV-14`, `PG-13`
     - `Older Kids (7-12)`: `TV-PG`, `PG`, `TV-Y7`, `TV-Y7-FV`
     - `Kids / All Ages (0-6)`: `TV-Y`, `TV-G`, `G`
  2. **Step 2: Training Datasets:** Encodes features via `ColumnTransformer` (TF-IDF for genres, One-Hot for format & country, scaler for year) and applies stratified 80/20 splitting.
  3. **Step 3: Model Training & Tuning:** Trains Decision Tree and Random Forest baselines, then optimizes hyperparameters via **`GridSearchCV`** with 5-fold Stratified Cross-Validation (`n_estimators`, `max_depth`, `min_samples_split`, `criterion`).
  4. **Step 4: Multi-Class Evaluation:** Produces per-class precision, recall, macro/weighted F1 reports, confusion matrix heatmaps, and Gini feature importances.
  5. **Step 5: Comparison & Interactive Form:** Benchmark table comparing baseline vs tuned models with an interactive category predictor.

---

### Task 4: Netflix Content Segmentation (Clustering)
* **Goal:** Group catalog titles into cohesive, actionable content clusters using unsupervised learning.
* **Workflow:**
  1. **Step 1: Feature Engineering:** Encodes hybrid multi-attribute data (TF-IDF for genres, One-Hot for format/rating/country, scaled year).
  2. **Step 2: Clustering Algorithms:** Evaluates $K = 2$ through $9$ using the **Elbow Method (Inertia/WCSS)** and **Silhouette Score Analysis**; trains final **K-Means** with optimal $K = 5$.
  3. **Step 3: Content Groups:** Attaches cluster labels to the catalog and audits cluster balance and volume shares.
  4. **Step 4: Visualizing Clusters:** Projects high-dimensional feature vectors to 2D via **Principal Component Analysis (PCA)**; generates scatter plots with marked centroids.
  5. **Step 5: Cluster Personas:** Profiles clusters by format ratio, top genres, and ratings to assign human-readable personas (*Kids & Family Animation*, *International TV Dramas*, *Docuseries & Real-World*, *Comedies*, *Action & Thrillers*); includes an interactive title cluster finder.

---

### Task 5: Netflix Trend Forecasting Model
* **Goal:** Forecast future platform content addition volume from historical time series patterns.
* **Workflow:**
  1. **Step 1: Time Features:** Parses `date_added`, extracts temporal components (year, month, quarter, day-of-week), and aggregates regular continuous monthly additions (2015–2021) with zero-gap filling.
  2. **Step 2: Historical Patterns:** Visualizes 6-month moving averages, verifies Friday release dominance, and decomposes the series into **Trend**, **12-Month Seasonality**, and **Residuals**.
  3. **Step 3: Forecasting Models:** Constructs autoregressive lag features ($t-1, t-2, t-3, t-6, t-12$), rolling statistics ($MA_3, MA_6$), and cyclical calendar features. Trains **Holt-Winters Exponential Smoothing**, **Random Forest Regressor**, and **Gradient Boosting Regressor**.
  4. **Step 4: Future Projections:** Generates out-of-sample forward forecasts for the next 12 to 24 months with $\pm 15\%$ confidence intervals.
  5. **Step 5: Accuracy Evaluation:** Evaluates test predictions using **MAE (Mean Absolute Error in titles)**, **RMSE**, **MAPE (%)**, and **$R^2$ Score**; includes an interactive horizon forecast viewer.

---

### Task 6: Netflix Content Success Analytics Engine
* **Goal:** Develop an enterprise-grade predictive engine and business intelligence solution to forecast the high-impact success potential of pre-release content.
* **Workflow:**
  1. **Step 1: Advanced Feature Engineering:**
     - **Acquisition Lag (`vintage_gap_years`):** Tracks delivery latency ($\text{year\_added} - \text{release\_year}$).
     - **Standardized Duration:** Numeric runtimes (minutes for movies, seasons for TV series).
     - **Director Prominence:** Catalog frequency weight.
     - **Co-Production Reach:** Cross-border multi-country financing indicator.
     - **Composite Content Impact Score (0–100):** Formulates success potential and defines the binary target `is_high_impact`.
  2. **Step 2: ML Pipeline Architecture:** Modular Scikit-Learn `Pipeline` with automated `ColumnTransformer` (`RobustScaler`, `TfidfVectorizer`, `OneHotEncoder`); trains **Logistic Regression**, **Random Forest**, and **Gradient Boosting** with 5-fold cross-validation.
  3. **Step 3: Model Comparison:** Benchmarks ROC-AUC, Precision, Recall, F1, and plots side-by-side ROC and Precision-Recall curves.
  4. **Step 4: Automated Business Intelligence:** AI rule engine computes strategic takeaways:
     - *Acquisition Timing ROI* (Day-and-date premiere uplift vs. vintage library acquisitions).
     - *Movie Runtime Optimization* (Peak viewer engagement window: 80–110 minutes).
     - *International Co-Production Reach Multiplier*.
     - *TV Franchise Longevity* (Renewal retention impact).
  5. **Step 5: Executive Dashboard & Simulator:** 4-panel Business Intelligence Dashboard and an interactive **Pre-Release Title Simulator (`@param`)** for content acquisition decision support.

---

## File Structure

```
├── Dataset.csv                             # Netflix catalog dataset (8,790+ titles)
├── Netflix_Recommendation_System.ipynb     # Task 1: Content Recommendation Engine
├── Content_Type_Prediction.ipynb           # Task 2: Content Type Classification
├── Audience_Rating_Classification.ipynb    # Task 3: Audience Rating Classification
├── Content_Segmentation_Clustering.ipynb   # Task 4: Content Segmentation (K-Means & PCA)
├── Netflix_Trend_Forecasting.ipynb         # Task 5: Time Series Trend Forecasting
├── Content_Success_Analytics_Engine.ipynb  # Task 6: Content Success Analytics & BI Engine
└── README.md                               # Project documentation
```

---

## How to Run the Notebooks

### In Google Colab (Recommended)
1. Navigate to [Google Colab](https://colab.research.google.com/).
2. Select **Upload** and upload any of the `.ipynb` files from the repository.
3. In the left sidebar, click the **Files** folder icon and upload `Dataset.csv` (or use the built-in Colab file upload prompt in the second cell of each notebook).
4. From the top menu, select **Runtime** $\to$ **Run all** (or execute cells sequentially using `Shift + Enter`).
5. Use the interactive form at the bottom of each notebook to test predictions, explore clusters, view forecasts, or simulate new title releases.

### In VS Code / JupyterLab
1. Open the project folder in VS Code.
2. Install the official **Python** and **Jupyter** extensions if not already installed.
3. Open any `.ipynb` file. *(Note: If a notebook opens as raw JSON text, right-click the file tab $\to$ select **"Reopen Editor With..."** $\to$ choose **"Jupyter Notebook"**).*
4. Select your Python 3 environment kernel in the top-right corner.
5. Click **Run All** to execute all cells.

---

## Required Python Libraries

All notebooks use standard, battle-tested data science libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn statsmodels
```

---

## Key Highlights & Evaluation Criteria

- **Task Completion:** All 6 tasks implemented end-to-end according to the specified 5-step workflows.
- **Code Quality:** Modular, clean, well-commented Python code utilizing Scikit-Learn `Pipeline`, `ColumnTransformer`, and vectorized operations.
- **Preventing Target Leakage:** Task 2 deliberately excludes raw duration strings (`"Season"` vs `"min"`) to ensure models learn genuine content attributes.
- **Robust Error Handling:** Date parsing with fallback imputations and cross-version compatibility for feature name extraction.
- **Interactive Functionality:** Every notebook concludes with a Google Colab interactive form widget (`@param`) for real-time user testing.
- **Executive Presentation:** Task 6 features an automated Business Intelligence engine and multi-panel executive visual dashboard.
