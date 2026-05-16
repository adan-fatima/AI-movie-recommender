import pandas as pd
import numpy as np
import requests, io, zipfile
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

print("All libraries loaded!")

print("Downloading MovieLens 100K dataset...")
url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
r = requests.get(url)

with zipfile.ZipFile(io.BytesIO(r.content)) as z:
    with z.open("ml-100k/u.data") as f:
        ratings = pd.read_csv(f, sep='\t', names=['userId','movieId','rating','timestamp'])
    with z.open("ml-100k/u.item") as f:
        movies = pd.read_csv(f, sep='|', encoding='latin-1',
            names=['movieId','title','release_date','video_release_date','IMDb_URL',
                   'unknown','Action','Adventure','Animation','Children','Comedy',
                   'Crime','Documentary','Drama','Fantasy','Film-Noir','Horror',
                   'Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western'],
            usecols=range(24))

GENRE_COLS = ['Action','Adventure','Animation','Children','Comedy','Crime',
              'Documentary','Drama','Fantasy','Film-Noir','Horror','Musical',
              'Mystery','Romance','Sci-Fi','Thriller','War','Western']

print("Injecting Bollywood and Modern Movie Datasets...")
bollywood_data = [
    {"title": "3 Idiots", "release_date": "25-Dec-2009", "Comedy": 1, "Drama": 1},
    {"title": "Dangal", "release_date": "23-Dec-2016", "Drama": 1},
    {"title": "Sholay", "release_date": "15-Aug-1975", "Action": 1, "Adventure": 1, "Drama": 1},
    {"title": "Stree", "release_date": "31-Aug-2018", "Horror": 1, "Comedy": 1},
    {"title": "Tumbbad", "release_date": "12-Oct-2018", "Horror": 1, "Fantasy": 1, "Thriller": 1},
    {"title": "Dilwale Dulhania Le Jayenge", "release_date": "20-Oct-1995", "Comedy": 1, "Drama": 1, "Romance": 1},
    {"title": "Lagaan", "release_date": "15-Jun-2001", "Drama": 1, "Musical": 1, "Romance": 1},
    {"title": "Andhadhun", "release_date": "05-Oct-2018", "Comedy": 1, "Crime": 1, "Thriller": 1},
    {"title": "Gangs of Wasseypur", "release_date": "22-Jun-2012", "Action": 1, "Crime": 1, "Thriller": 1},
    {"title": "Bhool Bhulaiyaa", "release_date": "12-Oct-2007", "Horror": 1, "Comedy": 1, "Mystery": 1}
]

latest_data = [
    {"title": "Dune: Part Two", "release_date": "01-Mar-2024", "Action": 1, "Adventure": 1, "Sci-Fi": 1},
    {"title": "Oppenheimer", "release_date": "21-Jul-2023", "Drama": 1, "Thriller": 1, "War": 1},
    {"title": "The Conjuring", "release_date": "19-Jul-2013", "Horror": 1, "Thriller": 1},
    {"title": "Hereditary", "release_date": "08-Jun-2018", "Horror": 1, "Mystery": 1},
    {"title": "Get Out", "release_date": "24-Feb-2017", "Horror": 1, "Mystery": 1, "Thriller": 1},
    {"title": "Everything Everywhere All at Once", "release_date": "25-Mar-2022", "Action": 1, "Comedy": 1, "Sci-Fi": 1},
    {"title": "Spider-Man: Across the Spider-Verse", "release_date": "02-Jun-2023", "Action": 1, "Adventure": 1, "Animation": 1},
    {"title": "Interstellar", "release_date": "07-Nov-2014", "Adventure": 1, "Drama": 1, "Sci-Fi": 1},
    {"title": "Parasite", "release_date": "30-May-2019", "Comedy": 1, "Drama": 1, "Thriller": 1},
    {"title": "The Dark Knight", "release_date": "18-Jul-2008", "Action": 1, "Crime": 1, "Thriller": 1}
]

start_id = movies['movieId'].max() + 1
additional_rows = []
additional_ratings = []

for idx, movie in enumerate(bollywood_data + latest_data):
    current_id = start_id + idx
    row = {col: 0 for col in movies.columns}
    row['movieId'] = current_id
    row['title'] = movie['title']
    row['release_date'] = movie['release_date']
    for g in GENRE_COLS:
        if g in movie:
            row[g] = movie[g]
    additional_rows.append(row)
    
    np.random.seed(current_id) 
    num_votes = int(np.random.randint(150, 400)) 
    simulated_ratings = np.random.normal(loc=4.4, scale=0.5, size=num_votes)
    simulated_ratings = np.clip(simulated_ratings, 1.0, 5.0)
    for rating in simulated_ratings:
        additional_ratings.append({
            'userId': np.random.randint(1, 943), 'movieId': current_id,
            'rating': round(rating), 'timestamp': 881250949
        })

movies = pd.concat([movies, pd.DataFrame(additional_rows)], ignore_index=True)
ratings = pd.concat([ratings, pd.DataFrame(additional_ratings)], ignore_index=True)

movies['genre_str'] = movies[GENRE_COLS].apply(lambda r: ' '.join([g for g, v in zip(GENRE_COLS, r) if v == 1]), axis=1)
movies['year'] = pd.to_numeric(movies['release_date'].astype(str).str.extract(r'(\d{4})')[0], errors='coerce').fillna(0).astype(int)

stats = ratings.groupby('movieId').agg(avg_rating=('rating','mean'), num_ratings=('rating','count')).reset_index()
movies = movies.drop(columns=['avg_rating', 'num_ratings'], errors='ignore').merge(stats, on='movieId', how='left')
movies['avg_rating'] = movies['avg_rating'].fillna(3.0)
movies['num_ratings'] = movies['num_ratings'].fillna(0)

C = movies['avg_rating'].mean()
m = movies['num_ratings'].quantile(0.25)
movies['score'] = ((movies['num_ratings'] / (movies['num_ratings'] + m)) * movies['avg_rating'] + (m / (movies['num_ratings'] + m)) * C)

print(f"Loaded {len(movies)} movies across ALL datasets!")

print("Training TF-IDF on genres...")
tfidf = TfidfVectorizer(ngram_range=(1, 2))
tfidf_matrix = tfidf.fit_transform(movies['genre_str'].fillna(''))
movie_cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

print("Training KNN model...")
knn_model = NearestNeighbors(n_neighbors=20, metric='cosine', algorithm='brute')
knn_model.fit(tfidf_matrix)

ERA_MAP = {
    "Any era":            (1900, 2026),
    "Classic (pre-1990)": (1900, 1989),
    "90s & 2000s":        (1990, 2009),
    "Modern (2010+)":     (2010, 2026),
}

MOOD_GENRE_MAP = {
    "Happy / Excited":    {"Comedy":1.0, "Animation":0.8, "Adventure":0.7, "Action":0.5},
    "Sad / Emotional":    {"Drama":1.0, "Romance":0.8, "Musical":0.5},
    "Stressed / Anxious": {"Comedy":0.9, "Animation":0.7, "Children":0.5},
    "Bored":              {"Action":1.0, "Adventure":0.9, "Sci-Fi":0.8, "Thriller":0.7},
    "Romantic":           {"Romance":1.0, "Drama":0.7, "Musical":0.5},
    "Scared / Thrilled":  {"Horror":1.0, "Thriller":0.9, "Mystery":0.7},
    "Inspired":           {"Documentary":1.0, "Drama":0.8, "War":0.5},
    "Adventurous":        {"Adventure":1.0, "Action":0.8, "Fantasy":0.7, "Sci-Fi":0.6},
    "Nostalgic":          {"Animation":0.8, "Musical":0.7, "Children":0.6, "Drama":0.5},
    "Relaxed / Chill":    {"Comedy":0.8, "Romance":0.6, "Musical":0.5, "Drama":0.4},
}

AUDIENCE_MAP = {
    "Alone":              {},
    "With partner/date": {"Romance": 0.5, "Drama": 0.3},
    "With family":        {"Animation": 0.7, "Children": 0.8, "Comedy": 0.4},
    "With friends":       {"Action": 0.5, "Comedy": 0.6, "Thriller": 0.4},
    "With kids":          {"Animation": 1.0, "Children": 1.0, "Comedy": 0.5},
}

def build_user_vector(mood, genre, audience):
    weights = {g: 0.0 for g in GENRE_COLS}
    for g, w in MOOD_GENRE_MAP.get(mood, {}).items():
        if g in weights:
            weights[g] += w
    if genre != "Any / Surprise me" and genre in weights:
        weights[genre] += 1.5
    for g, w in AUDIENCE_MAP.get(audience, {}).items():
        if g in weights:
            weights[g] += w
    return np.array([weights[g] for g in GENRE_COLS])

def recommend_movies(mood, genre, era, audience, n=5):
    year_min, year_max = ERA_MAP.get(era, (1900, 2026))
    if era == "Any era":
        filtered = movies.copy().reset_index(drop=True)
    else:
        filtered = movies[(movies['year'] >= year_min) & (movies['year'] <= year_max)].copy().reset_index(drop=True)

    if genre != "Any / Surprise me":
        filtered = filtered[filtered[genre] == 1].copy().reset_index(drop=True)

    if filtered.empty:
        raise ValueError(f"No movies found for '{era}' matching '{genre}'.")

    user_vec = build_user_vector(mood, genre, audience)
    genre_matrix = filtered[GENRE_COLS].values.astype(float)
    genre_sim = cosine_similarity(user_vec.reshape(1, -1), genre_matrix)[0]
    filtered['genre_sim'] = genre_sim

    scaler = MinMaxScaler()
    if len(filtered) > 1:
        filtered[['norm_sim', 'norm_score']] = scaler.fit_transform(filtered[['genre_sim', 'score']])
    else:
        filtered['norm_sim'] = 1.0
        filtered['norm_score'] = 1.0
        
    filtered['final_score'] = 0.65 * filtered['norm_sim'] + 0.35 * filtered['norm_score']
    result = filtered.nlargest(n, 'final_score')
    return result[['title','year','genre_str','avg_rating','num_ratings','genre_sim','final_score']].reset_index(drop=True)

def display_results(results, mood, genre, era, audience):
    mood_colors = {
        "Happy / Excited":    "#F59E0B", "Sad / Emotional":    "#6366F1",
        "Stressed / Anxious": "#10B981", "Bored":              "#F97316",
        "Romantic":           "#EC4899", "Scared / Thrilled":  "#EF4444",
        "Inspired":           "#8B5CF6", "Adventurous":        "#06B6D4",
        "Nostalgic":          "#84CC16", "Relaxed / Chill":    "#14B8A6",
    }
    accent = mood_colors.get(mood, "#F59E0B")
    cards  = ""

    for i, row in results.iterrows():
        stars_full  = int(round(row['avg_rating'] / 2))
        stars_empty = 5 - stars_full
        stars       = "&#9733;" * stars_full + "&#9734;" * stars_empty
        match_pct   = int(row['genre_sim'] * 100)
        score_pct   = int(row['final_score'] * 100)
        genres_disp = row['genre_str'].replace(' ', ' | ') if row['genre_str'] else 'N/A'
        year_disp   = str(int(row['year'])) if row['year'] > 0 else 'N/A'

        cards += f"""
        <div style="background:#1a1a2e;border-radius:14px;padding:22px;margin-bottom:18px;border-left:4px solid {accent};">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:8px;">
            <div>
              <span style="background:{accent};color:#0a0a1a;font-weight:700;font-size:11px;padding:3px 10px;border-radius:20px;">#{i+1}</span>
              <h3 style="margin:8px 0 4px;font-size:19px;color:#fff;font-family:Georgia,serif;">
                {row['title']} <span style="font-size:13px;color:#888;font-weight:400;">({year_disp})</span>
              </h3>
              <p style="margin:0;color:#aaa;font-size:12px;">{genres_disp}</p>
            </div>
            <div style="text-align:right;">
              <div style="color:{accent};font-size:16px;">{stars}</div>
              <div style="color:#888;font-size:12px;">{row['avg_rating']:.1f}/5 &middot; {int(row['num_ratings'])} votes</div>
            </div>
          </div>
          <div style="margin-top:14px;display:flex;gap:12px;flex-wrap:wrap;">
            <div style="flex:1;min-width:160px;">
              <p style="color:#888;font-size:11px;margin:0 0 4px;">GENRE MATCH (ML)</p>
              <div style="background:#0d0d1a;border-radius:6px;height:8px;overflow:hidden;">
                <div style="width:{match_pct}%;background:{accent};height:100%;border-radius:6px;"></div>
              </div>
              <p style="color:{accent};font-size:12px;margin:3px 0 0;">{match_pct}% match</p>
            </div>
            <div style="flex:1;min-width:160px;">
              <p style="color:#888;font-size:11px;margin:0 0 4px;">OVERALL SCORE</p>
              <div style="background:#0d0d1a;border-radius:6px;height:8px;overflow:hidden;">
                <div style="width:{score_pct}%;background:#6366F1;height:100%;border-radius:6px;"></div>
              </div>
              <p style="color:#6366F1;font-size:12px;margin:3px 0 0;">{score_pct}/100</p>
            </div>
          </div>
        </div>"""

    html = f"""
    <div style="font-family:'Segoe UI',sans-serif;max-width:760px;margin:0 auto;background:#0d0d1a;padding:28px;border-radius:20px;color:#fff;">
      <div style="text-align:center;margin-bottom:24px;">
        <h1 style="font-family:Georgia,serif;font-size:26px;margin:0 0 6px;color:{accent};">Your ML Movie Picks</h1>
        <p style="color:#888;font-size:13px;margin:0;">
          Mood: <strong style="color:{accent};">{mood}</strong> &nbsp;&middot;&nbsp;
          Genre: <strong style="color:{accent};">{genre}</strong> &nbsp;&middot;&nbsp;
          Era: <strong style="color:{accent};">{era}</strong> &nbsp;&middot;&nbsp;
          With: <strong style="color:{accent};">{audience}</strong>
        </p>
      </div>
      {cards}
    </div>"""
    display(HTML(html))

def run_recommender():
    clear_output(wait=True)
    print("ALL-ERA ML MOVIE RECOMMENDER (Fixed Filter)")
    
    style  = {'description_width': '160px'}
    layout = widgets.Layout(width='480px')

    mood_w = widgets.Dropdown(options=list(MOOD_GENRE_MAP.keys()), value="Happy / Excited", description="Your mood:", style=style, layout=layout)
    genre_w = widgets.Dropdown(options=["Any / Surprise me"] + GENRE_COLS, value="Any / Surprise me", description="Preferred genre:", style=style, layout=layout)
    era_w = widgets.Dropdown(options=list(ERA_MAP.keys()), value="Any era", description="Movie era:", style=style, layout=layout)
    audience_w = widgets.Dropdown(options=list(AUDIENCE_MAP.keys()), value="Alone", description="Watching with:", style=style, layout=layout)
    
    button = widgets.Button(description="Recommend Movies", button_style='success', layout=widgets.Layout(width='200px', height='40px', margin='14px 0 0 164px'))
    out = widgets.Output()

    display(mood_w, genre_w, era_w, audience_w, button, out)

    def on_click(b):
        with out:
            clear_output(wait=True)
            try:
                results = recommend_movies(mood=mood_w.value, genre=genre_w.value, era=era_w.value, audience=audience_w.value, n=5)
                clear_output(wait=True)
                display_results(results, mood_w.value, genre_w.value, era_w.value, audience_w.value)
            except ValueError as e:
                print(f"Filter error: {e}")
            except Exception as e:
                print(f"Error: {e}")

    button.on_click(on_click)

run_recommender()
