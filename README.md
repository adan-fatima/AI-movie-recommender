# ML Movie Recommendation System

An interactive machine learning-based movie recommender system built in Python.
It suggests personalized movies based on mood, genre, era, and viewing audience using a hybrid of content-based filtering and rating-based scoring.

---

## Features

* Personalized recommendations based on:

  * Mood (Happy, Sad, Excited, etc.)
  * Genre preference
  * Movie era (Classic, 90s, Modern, etc.)
  * Viewing audience (alone, family, friends, etc.)
* Uses real MovieLens 100K dataset
* Includes added Bollywood classics
* Includes modern global blockbuster movies (up to 2026)
* Hybrid recommendation system:

  * Content-based filtering (Cosine Similarity)
  * Bayesian average rating system
  * Weighted final ranking model
* Interactive UI using ipywidgets in Jupyter Notebook
* HTML-based result cards with visual score bars

---

## How It Works

### 1. Data Preparation

* Loads MovieLens 100K dataset
* Adds Bollywood and modern movies manually
* Generates simulated ratings for new movies

### 2. Feature Engineering

* Converts genres into text format
* Extracts movie release year
* Creates rating statistics (average rating, number of votes)
* Computes a Bayesian weighted score

### 3. Recommendation Engine

* Builds a user preference vector from:

  * Mood
  * Genre
  * Audience type
* Uses Cosine Similarity to match movies
* Combines:

  * 65% genre similarity
  * 35% popularity score
* Ranks top 5 movies

### 4. Interactive UI

* Dropdown menus for user input
* One-click recommendation button
* Styled movie cards showing:

  * Rating stars
  * Match percentage
  * Final score bar

---

## Tech Stack

* Python
* Pandas and NumPy
* Scikit-learn (TF-IDF, Cosine Similarity, KNN, Scaling)
* ipywidgets (UI)
* HTML and CSS (for visualization in notebook)
* MovieLens Dataset

---

## Dataset Used

MovieLens 100K dataset
[https://grouplens.org/datasets/movielens/100k/](https://grouplens.org/datasets/movielens/100k/)

Additional custom data:

* Bollywood classic movies
* Modern global movies

---

## How to Run

### Option 1: Google Colab

1. Open Google Colab
2. Paste the full code
3. Run all cells
4. Run:

```python
run_recommender()
```

---

### Option 2: Local Jupyter Notebook

```bash
pip install pandas numpy scikit-learn ipywidgets requests
jupyter notebook
```

Then run:

```python
run_recommender()
```

---

## Algorithms Used

* Cosine Similarity for content-based matching
* TF-IDF vectorization for genre representation
* MinMax Scaling for normalization
* Bayesian average rating for fair scoring
* Hybrid scoring system:

```
Final Score = 65% Genre Similarity + 35% Popularity Score
```

---

## Example Use Case

* Mood: Bored
* Genre: Action
* Era: Modern (2010+)
* Audience: With friends

Output: Top 5 action-focused modern movies with high relevance and rating balance

---

## Future Improvements

* Add Netflix or IMDb API integration
* Implement collaborative filtering
* Deploy as a web application using Flask or Streamlit
* Add user login and history tracking
* Improve UI into a full dashboard

---

## Author

Built by a software engineering student focused on machine learning, recommender systems, and full-stack AI applications.
