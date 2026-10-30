# Task 1 (Easy) - Netflix Content Recommendation System
**AUSPIFY - Bring Future Through Tech**

An end-to-end Machine Learning Content-Based Recommendation System that suggests similar Netflix titles using genres, categories, directors, country of origin, and content attributes.

---

## Task Workflow Implementation

### Step 1: Prepare Content-Related Features
- **Data Ingestion:** Loaded 8,790 Netflix titles (`Dataset.csv`).
- **Data Cleaning:** Handled missing values and removed placeholder texts like `"Not Given"`.
- **Feature Tokenization:**
  - Multi-word director and country names are sanitized into unified tokens (e.g., `Steven Spielberg` $\to$ `stevenspielberg`) to prevent false cross-matching on common names.
  - Multi-word genres (e.g., `Action & Adventure` $\to$ `actionadventure`) are cleaned and boosted.
- **Metadata Soup Creation:** Combined `type`, `listed_in`, `director`, `country`, and `rating` into a single text representation for each title:
  $$\text{Feature Soup} = \text{type} + 2 \times \text{listed\_in} + \text{director} + \text{country} + \text{rating}$$

### Step 2: Convert Text Data into Machine-Readable Format
- Used **TF-IDF Vectorizer** (`sklearn.feature_extraction.text.TfidfVectorizer`) with sublinear term-frequency scaling and English stop-words filtering.
- Resulting TF-IDF Matrix: **8,790 titles $\times$ 5,126 feature tokens**.

### Step 3: Calculate Content Similarity Scores
- Calculated pairwise **Cosine Similarity**:
  $$\text{Cosine Similarity}(\vec{u}, \vec{v}) = \frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\| \|\vec{v}\|}$$
- Produced an $8,790 \times 8,790$ symmetric similarity matrix capturing semantic alignment between every pair of titles.

### Step 4: Generate Recommendations for Selected Titles
- Look up title with case-insensitive search and fallback substring matching.
- Rank similarity scores descending, exclude self-similarity, and return the top $N$ titles with match percentages.

### Step 5: Evaluate Recommendation Quality
- **Mean Genre Overlap (Jaccard Index):** **0.820 / 1.000**
  $$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$
- **Content Type Consistency Rate:** **100.0%** (Movies consistently recommend Movies; TV Shows recommend TV Shows).
- **Mean Recommendation Similarity Score:** **79.7%**

---

## How to Run

1. Run the recommender script:
   ```bash
   python recommender.py
   ```

2. Interactive recommendations in Python:
   ```python
   from recommender import NetflixRecommender

   recommender = NetflixRecommender("Dataset.csv")
   recommender.load_and_preprocess()
   recommender.vectorize_features()
   recommender.calculate_similarity()

   # Get recommendations for any title
   recs = recommender.get_recommendations("Squid Game", top_n=5)
   print(recs[['title', 'type', 'listed_in', 'similarity_score']])
   ```
