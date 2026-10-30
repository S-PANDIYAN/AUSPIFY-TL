# Netflix Content Recommendation System

A Machine Learning-based Content Recommendation System that suggests similar Netflix movies and TV shows based on genres, categories, directors, country of origin, and content attributes.

---

## Task Workflow

### Step 1: Prepare Content-Related Features
- **Data Ingestion:** Loaded Netflix titles from `Dataset.csv`.
- **Data Cleaning:** Handled missing values and removed placeholder texts like `'Not Given'`.
- **Feature Tokenization:**
  - Multi-word director and country names are sanitized into unified tokens (e.g., `Steven Spielberg` $\to$ `stevenspielberg`) to prevent false cross-matching on common first/last names.
  - Multi-word genres (e.g., `Action & Adventure` $\to$ `actionadventure`) are cleaned and boosted.
- **Metadata Soup Creation:** Combined `type`, `listed_in`, `director`, `country`, and `rating` into a single unified text representation for each title:
  $$\text{Feature Soup} = \text{type} + 2 \times \text{listed\_in} + \text{director} + \text{country} + \text{rating}$$

---

### Step 2: Convert Text Data into Machine-Readable Format
- Implemented **TF-IDF Vectorizer** (`sklearn.feature_extraction.text.TfidfVectorizer`) with sublinear term-frequency scaling and English stop-words filtering.
- Resulting TF-IDF feature space maps each title into a high-dimensional vector based on content attributes.

---

### Step 3: Calculate Content Similarity Scores
- Calculated pairwise **Cosine Similarity**:
  $$\text{Cosine Similarity}(\vec{u}, \vec{v}) = \frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\| \|\vec{v}\|}$$
- Constructed an $N \times N$ symmetric similarity matrix measuring semantic and attribute closeness between all titles.

---

### Step 4: Generate Recommendations for Selected Titles
- Look up titles using case-insensitive search with partial/fuzzy match fallback.
- Retrieve pairwise similarity scores, sort in descending order, exclude self-match, and return top-$N$ similar titles with match percentage scores.

---

### Step 5: Evaluate Recommendation Quality
- **Genre Overlap (Jaccard Index):** Evaluates how closely recommended titles match the genres of the query title:
  $$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$
- **Content Type Consistency Rate:** Measures format alignment (Movies recommending Movies, TV Shows recommending TV Shows).
- **Mean Similarity Confidence Score:** Quantifies average model confidence across recommendations.

---

## File Structure

```
├── Dataset.csv                          # Netflix catalog dataset
├── Netflix_Recommendation_System.ipynb  # Google Colab / Jupyter Notebook
└── README.md                            # Project documentation
```

---

## How to Run in Google Colab

1. Open [Google Colab](https://colab.research.google.com/).
2. Click **Upload** and upload `Netflix_Recommendation_System.ipynb`.
3. Upload `Dataset.csv` into the Colab file explorer (or use the built-in upload prompt in Cell 2).
4. Run each cell sequentially from **Step 1** through **Step 5**.
5. Use the interactive form at the bottom of the notebook to type any title (e.g., `Squid Game`, `Jaws`, `Grown Ups`) and receive instant recommendations.
