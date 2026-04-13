"""
CineMatch — Flask Application
All routes and startup logic. Algorithm logic lives in algorithms/.
"""

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import sqlite3
import os
import math

from algorithms.recommender import ContentRecommender
from algorithms.kmeans_cluster import run_kmeans
from algorithms.apriori_rules import run_apriori
from algorithms.knn_recommend import KNNRecommender

app = Flask(__name__)
app.secret_key = os.environ.get('SESSION_SECRET', 'cinematch_secret_key_2024')

# ---------------------------------------------------------------------------
# Load data ONCE at startup
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'movies_clean.csv'))

# Ensure movie_id is int
df['movie_id'] = pd.to_numeric(df['movie_id'], errors='coerce').fillna(0).astype(int)

# Init algorithms ONCE at startup
recommender = ContentRecommender(df)
knn = KNNRecommender(df)

# Pre-compute discover data (cached in modules)
_apriori_rules = None
_clusters = None


def get_db():
    """Return a SQLite connection, creating the DB and tables if needed."""
    db_path = os.path.join(BASE_DIR, 'database', 'users.db')
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS favorites (
            username TEXT,
            movie_id INTEGER,
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (username, movie_id)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS watchlist (
            username TEXT,
            movie_id INTEGER,
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (username, movie_id)
        )
    """)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS ratings (
        username TEXT,
        movie_id INTEGER,
        rating FLOAT,
        rated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (username, movie_id)
    )
""")
    conn.commit()
    return conn

def generate_movie_summary(movie):
    def clean(val):
        return val if val and str(val) not in ('Unknown', 'nan', '', 'None') else None

    name     = clean(movie.get('name', ''))
    genre    = clean(movie.get('genre', ''))
    score    = movie.get('score', 0)
    director = clean(movie.get('director', ''))
    writer   = clean(movie.get('writer', ''))
    star     = clean(movie.get('star', ''))
    country  = clean(movie.get('country', ''))
    year     = movie.get('year', 0)
    runtime  = movie.get('runtime', 0)
    votes    = movie.get('votes', 0)
    classification = clean(movie.get('classification', ''))
    rating   = clean(movie.get('rating', ''))

    if votes >= 1_000_000:
        votes_str = f"{votes/1_000_000:.1f}M"
    elif votes >= 1_000:
        votes_str = f"{votes/1_000:.0f}K"
    else:
        votes_str = str(votes)

    summary = f"{name} is a {genre} film" if genre else f"{name} is a film"
    if year:
        summary += f" released in {year}"
    if country:
        summary += f", produced in {country}"
    summary += ". "

    if director:
        summary += f"Directed by {director}"
        if writer and writer != director:
            summary += f" and written by {writer}"
        summary += ". "

    if star:
        summary += f"Starring {star}. "

    if runtime:
        summary += f"The film runs for {runtime} minutes"
        if rating:
            summary += f" and is rated {rating}"
        summary += ". "

    summary += f"It holds a score of {score}/10"
    if votes_str:
        summary += f" based on {votes_str} votes"
    if classification:
        summary += f", classified as a {classification} film"
    summary += "."

    return summary

def movie_to_dict(row):
    """Convert a DataFrame row to a clean JSON-serializable dict."""
    return {
        'movie_id': int(row['movie_id']),
        'name': str(row['name']) if pd.notna(row['name']) else '',
        'genre': str(row['genre']) if pd.notna(row['genre']) else '',
        'rating': str(row['rating']) if pd.notna(row['rating']) else '',
        'year': int(row['year']) if pd.notna(row['year']) else 0,
        'score': round(float(row['score']), 1) if pd.notna(row['score']) else 0.0,
        'weighted_score': round(float(row['weighted_score']), 2) if pd.notna(row['weighted_score']) else 0.0,
        'votes': int(row['votes']) if pd.notna(row['votes']) else 0,
        'director': str(row['director']) if pd.notna(row['director']) else '',
        'writer': str(row['writer']) if pd.notna(row['writer']) else '',
        'star': str(row['star']) if pd.notna(row['star']) else '',
        'country': str(row['country']) if pd.notna(row['country']) else '',
        'budget': int(row['budget']) if pd.notna(row['budget']) and row['budget'] != 0 else 0,
        'gross': int(row['gross']) if pd.notna(row['gross']) and row['gross'] != 0 else 0,
        'company': str(row['company']) if pd.notna(row['company']) else '',
        'runtime': int(row['runtime']) if pd.notna(row['runtime']) else 0,
        'classification': str(row['classification']) if pd.notna(row['classification']) else '',
        'released': str(row['released']) if pd.notna(row['released']) else '',
    }


# ---------------------------------------------------------------------------
# Page Routes
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    total_movies = len(df)
    genres_count = df['genre'].nunique()
    top_movie = df.nlargest(1, 'weighted_score').iloc[0]
    genres = sorted(df['genre'].dropna().unique().tolist())
    trending = df.nlargest(10, 'weighted_score').to_dict('records')
    trending = [movie_to_dict(pd.Series(m)) for m in trending]
    return render_template('index.html',
                           total_movies=total_movies,
                           genres_count=genres_count,
                           top_movie=movie_to_dict(top_movie),
                           genres=genres,
                           trending=trending)


@app.route('/browse')
def browse():
    genres = sorted(df['genre'].dropna().unique().tolist())
    countries = sorted(df['country'].dropna().unique().tolist())
    min_year = int(df['year'].min())
    max_year = int(df['year'].max())
    return render_template('browse.html',
                           genres=genres,
                           countries=countries,
                           min_year=min_year,
                           max_year=max_year)


@app.route('/movie/<int:movie_id>')
def movie_detail(movie_id):
    movie_row = df[df['movie_id'] == movie_id]
    if movie_row.empty:
        return render_template('404.html'), 404
    movie = movie_to_dict(movie_row.iloc[0])
    summary = generate_movie_summary(movie)
    similar = recommender.get_similar(movie_id, top_n=6)
    genre = movie['genre']
    also_like = df[(df['genre'] == genre) & (df['movie_id'] != movie_id)].nlargest(4, 'weighted_score')
    also_like = [movie_to_dict(r) for _, r in also_like.iterrows()]
    username = session.get('username', '')
    in_favorites = False
    in_watchlist = False
    user_rating = None
    if username:
        conn = get_db()
        fav = conn.execute('SELECT 1 FROM favorites WHERE username=? AND movie_id=?', (username, movie_id)).fetchone()
        watch = conn.execute('SELECT 1 FROM watchlist WHERE username=? AND movie_id=?', (username, movie_id)).fetchone()
        rat = conn.execute('SELECT rating FROM ratings WHERE username=? AND movie_id=?', (username, movie_id)).fetchone()
        conn.close()
        in_favorites = fav is not None
        in_watchlist = watch is not None
        user_rating = float(rat['rating']) if rat else None
    return render_template('movie.html',
                           movie=movie,
                           summary=summary,
                           similar=similar,
                           also_like=also_like,
                           in_favorites=in_favorites,
                           in_watchlist=in_watchlist,
                           username=username,
                           user_rating=user_rating)

@app.route('/discover')
def discover():
    genres = df['genre'].dropna()
    genre_counts = genres.value_counts().to_dict()
    genre_avg_score = df.groupby('genre')['score'].mean().round(2).to_dict()
    classification_counts = df['classification'].value_counts().to_dict()
    most_popular_genre = genres.value_counts().index[0]
    highest_rated_genre = df.groupby('genre')['score'].mean().idxmax()
    most_votes_genre = df.groupby('genre')['votes'].sum().idxmax()

    # Score histogram
    bins = [1,2,3,4,5,6,7,8,9,10]
    score_hist = []
    for i in range(len(bins)-1):
        count = int(((df['score'] >= bins[i]) & (df['score'] < bins[i+1])).sum())
        score_hist.append(count)
    score_hist[-1] += int((df['score'] == 10).sum())

    return render_template('discover.html',
                           genre_counts=genre_counts,
                           genre_avg_score=genre_avg_score,
                           classification_counts=classification_counts,
                           most_popular_genre=most_popular_genre,
                           highest_rated_genre=highest_rated_genre,
                           most_votes_genre=most_votes_genre,
                           score_hist=score_hist)

@app.route('/recommend')
def recommend_page():
    # Pass all movie names/ids for the multi-select
    movies_list = df[['movie_id', 'name', 'genre', 'year']].sort_values('name').to_dict('records')
    for m in movies_list:
        m['movie_id'] = int(m['movie_id'])
        m['year'] = int(m['year']) if pd.notna(m['year']) else 0
    return render_template('recommend.html', movies_list=movies_list)


@app.route('/profile')
def profile():
    username = session.get('username')
    if not username:
        return redirect(url_for('login'))
    return render_template('profile.html', username=username)


# ---------------------------------------------------------------------------
# Auth Routes
# ---------------------------------------------------------------------------

@app.route('/register', methods=['GET', 'POST'])
def register():
    if session.get('username'):
        return redirect(url_for('profile'))
    error = None
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        confirm = request.form.get('confirm_password', '')
        if not username:
            error = 'Username is required.'
        elif len(password) < 6:
            error = 'Password must be at least 6 characters.'
        elif password != confirm:
            error = 'Passwords do not match.'
        else:
            conn = get_db()
            existing = conn.execute('SELECT 1 FROM users WHERE username=?', (username,)).fetchone()
            if existing:
                error = 'Username is already taken.'
                conn.close()
            else:
                conn.execute('INSERT INTO users (username, password_hash) VALUES (?, ?)',
                             (username, generate_password_hash(password)))
                conn.commit()
                conn.close()
                session['username'] = username
                return redirect(url_for('profile'))
    return render_template('register.html', error=error)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if session.get('username'):
        return redirect(url_for('profile'))
    error = None
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        conn = get_db()
        user = conn.execute('SELECT * FROM users WHERE username=?', (username,)).fetchone()
        conn.close()
        if user and check_password_hash(user['password_hash'], password):
            session['username'] = username
            return redirect(url_for('profile'))
        else:
            error = 'Invalid username or password.'
    return render_template('login.html', error=error)


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------

@app.route('/data/movies')
def api_movies():
    """Filtered + paginated movie list."""
    filtered = df.copy()
    genre = request.args.get('genre', '')
    country = request.args.get('country', '')
    classification = request.args.get('classification', '')
    search = request.args.get('search', '')
    year_min = request.args.get('year_min', type=int)
    year_max = request.args.get('year_max', type=int)
    score_min = request.args.get('score_min', type=float)
    score_max = request.args.get('score_max', type=float)
    sort_by = request.args.get('sort', 'weighted_score')
    page = request.args.get('page', 1, type=int)
    per_page = 20

    if genre:
        filtered = filtered[filtered['genre'] == genre]
    if country:
        filtered = filtered[filtered['country'] == country]
    if classification:
        filtered = filtered[filtered['classification'] == classification]
    if search:
        mask = (
            filtered['name'].str.contains(search, case=False, na=False) |
            filtered['star'].str.contains(search, case=False, na=False) |
            filtered['director'].str.contains(search, case=False, na=False)
        )
        filtered = filtered[mask]
    if year_min:
        filtered = filtered[filtered['year'] >= year_min]
    if year_max:
        filtered = filtered[filtered['year'] <= year_max]
    if score_min is not None:
        filtered = filtered[filtered['score'] >= score_min]
    if score_max is not None:
        filtered = filtered[filtered['score'] <= score_max]

    valid_sorts = ['score', 'weighted_score', 'year', 'votes']
    if sort_by not in valid_sorts:
        sort_by = 'weighted_score'
    filtered = filtered.sort_values(sort_by, ascending=False)

    total = len(filtered)
    total_pages = max(1, math.ceil(total / per_page))
    start = (page - 1) * per_page
    end = start + per_page
    page_data = filtered.iloc[start:end]
    movies = [movie_to_dict(row) for _, row in page_data.iterrows()]
    return jsonify({'movies': movies, 'total': total, 'page': page, 'total_pages': total_pages})


@app.route('/data/similar/<int:movie_id>')
def api_similar(movie_id):
    similar = recommender.get_similar(movie_id, top_n=6)
    return jsonify({'similar': similar})


@app.route('/data/clusters')
def api_clusters():
    clusters = run_kmeans(df)
    return jsonify({'clusters': clusters})


@app.route('/data/rules')
def api_rules():
    rules = run_apriori(df)
    return jsonify({'rules': rules})


@app.route('/data/recommend', methods=['POST'])
def api_recommend():
    data = request.get_json()
    movie_ids = [int(mid) for mid in data.get('movie_ids', [])]
    if not movie_ids:
        return jsonify({'content_based': [], 'collaborative': []})
    content_recs = recommender.get_recommendations_for_multiple(movie_ids, top_n=10)
    collaborative_recs = knn.get_recommendations(movie_ids, top_n=10)
    return jsonify({'content_based': content_recs, 'collaborative': collaborative_recs})


@app.route('/data/search')
def api_search():
    query = request.args.get('q', '')
    if not query or len(query) < 2:
        return jsonify({'results': []})
    mask = (
        df['name'].str.contains(query, case=False, na=False) |
        df['star'].str.contains(query, case=False, na=False) |
        df['director'].str.contains(query, case=False, na=False)
    )
    results = df[mask].nlargest(10, 'weighted_score')
    return jsonify({'results': [movie_to_dict(r) for _, r in results.iterrows()]})


@app.route('/data/favorites/add', methods=['POST'])
def api_favorites_add():
    username = session.get('username')
    if not username:
        return jsonify({'error': 'login_required'}), 401
    data = request.get_json()
    movie_id = int(data.get('movie_id', 0))
    if not movie_id:
        return jsonify({'success': False, 'error': 'Missing movie_id'})
    conn = get_db()
    conn.execute('INSERT OR IGNORE INTO favorites (username, movie_id) VALUES (?, ?)', (username, movie_id))
    conn.commit()
    conn.close()
    return jsonify({'success': True})


@app.route('/data/favorites/remove', methods=['POST'])
def api_favorites_remove():
    data = request.get_json()
    username = data.get('username') or session.get('username', '')
    movie_id = int(data.get('movie_id', 0))
    if not username or not movie_id:
        return jsonify({'success': False})
    conn = get_db()
    conn.execute('DELETE FROM favorites WHERE username=? AND movie_id=?', (username, movie_id))
    conn.commit()
    conn.close()
    return jsonify({'success': True})


@app.route('/data/watchlist/add', methods=['POST'])
def api_watchlist_add():
    username = session.get('username')
    if not username:
        return jsonify({'error': 'login_required'}), 401
    data = request.get_json()
    movie_id = int(data.get('movie_id', 0))
    if not movie_id:
        return jsonify({'success': False, 'error': 'Missing movie_id'})
    conn = get_db()
    conn.execute('INSERT OR IGNORE INTO watchlist (username, movie_id) VALUES (?, ?)', (username, movie_id))
    conn.commit()
    conn.close()
    return jsonify({'success': True})


@app.route('/data/watchlist/remove', methods=['POST'])
def api_watchlist_remove():
    data = request.get_json()
    username = data.get('username') or session.get('username', '')
    movie_id = int(data.get('movie_id', 0))
    if not username or not movie_id:
        return jsonify({'success': False})
    conn = get_db()
    conn.execute('DELETE FROM watchlist WHERE username=? AND movie_id=?', (username, movie_id))
    conn.commit()
    conn.close()
    return jsonify({'success': True})


@app.route('/data/profile/<username>')
def api_profile(username):
    conn = get_db()
    fav_ids = [r[0] for r in conn.execute('SELECT movie_id FROM favorites WHERE username=? ORDER BY added_at DESC', (username,)).fetchall()]
    watch_ids = [r[0] for r in conn.execute('SELECT movie_id FROM watchlist WHERE username=? ORDER BY added_at DESC', (username,)).fetchall()]
    conn.close()
    favorites = []
    for mid in fav_ids:
        row = df[df['movie_id'] == mid]
        if not row.empty:
            favorites.append(movie_to_dict(row.iloc[0]))
    watchlist = []
    for mid in watch_ids:
        row = df[df['movie_id'] == mid]
        if not row.empty:
            watchlist.append(movie_to_dict(row.iloc[0]))
    fav_genre = ''
    avg_fav_score = 0.0
    if favorites:
        genres_list = [m['genre'] for m in favorites if m['genre']]
        if genres_list:
            from collections import Counter
            fav_genre = Counter(genres_list).most_common(1)[0][0]
        scores = [m['score'] for m in favorites if m['score']]
        if scores:
            avg_fav_score = round(sum(scores) / len(scores), 1)
    return jsonify({
        'favorites': favorites,
        'watchlist': watchlist,
        'fav_genre': fav_genre,
        'avg_fav_score': avg_fav_score
    })


@app.route('/data/set_username', methods=['POST'])
def api_set_username():
    data = request.get_json()
    username = data.get('username', '').strip()
    if username:
        session['username'] = username
        return jsonify({'success': True, 'username': username})
    return jsonify({'success': False})

@app.route('/data/ratings/add', methods=['POST'])
def api_ratings_add():
    username = session.get('username')
    if not username:
        return jsonify({'error': 'login_required'}), 401
    data = request.get_json()
    movie_id = int(data.get('movie_id', 0))
    rating = float(data.get('rating', 0))
    if not movie_id or rating < 1 or rating > 10:
        return jsonify({'success': False, 'error': 'Invalid data'})
    conn = get_db()
    conn.execute('INSERT OR REPLACE INTO ratings (username, movie_id, rating) VALUES (?, ?, ?)',
                 (username, movie_id, rating))
    conn.commit()
    conn.close()
    return jsonify({'success': True, 'rating': rating})


@app.route('/data/ratings/<username>')
def api_ratings(username):
    conn = get_db()
    rows = conn.execute('SELECT movie_id, rating FROM ratings WHERE username=? ORDER BY rated_at DESC',
                        (username,)).fetchall()
    conn.close()
    results = []
    for r in rows:
        row = df[df['movie_id'] == r['movie_id']]
        if not row.empty:
            m = movie_to_dict(row.iloc[0])
            m['user_rating'] = r['rating']
            m['diff'] = round(r['rating'] - m['score'], 1)
            results.append(m)
    avg = round(sum(r['user_rating'] for r in results) / len(results), 1) if results else 0
    return jsonify({'ratings': results, 'avg_rating': avg})

@app.route('/data/clusters/<int:cluster_id>/movies')
def api_cluster_movies(cluster_id):
    page = request.args.get('page', 1, type=int)
    per_page = 10
    clusters = run_kmeans(df)
    if cluster_id >= len(clusters):
        return jsonify({'movies': [], 'total': 0, 'total_pages': 0})
    cluster = clusters[cluster_id]
    all_ids = cluster.get('all_movie_ids', [])
    
    # Debug: log what we have
    print(f"Cluster {cluster_id}: {len(all_ids)} movies, first 3 ids: {all_ids[:3]}")
    print(f"DataFrame movie_id sample: {df['movie_id'].head(3).tolist()}")
    
    total = len(all_ids)
    start = (page - 1) * per_page
    page_ids = all_ids[start:start + per_page]
    movies = []
    for mid in page_ids:
        row = df[df['movie_id'] == int(mid)]
        if not row.empty:
            movies.append(movie_to_dict(row.iloc[0]))
    
    print(f"Found {len(movies)} movies for page {page}")
    
    return jsonify({
        'movies': movies,
        'total': total,
        'total_pages': math.ceil(total / per_page) if total > 0 else 0
    })
@app.route('/data/profile/recommend', methods=['POST'])
def api_profile_recommend():
    data = request.get_json()
    username = data.get('username') or session.get('username', '')
    if not username:
        return jsonify({'recommendations': []})
    conn = get_db()
    fav_ids = [r[0] for r in conn.execute(
        'SELECT movie_id FROM favorites WHERE username=?', (username,)).fetchall()]
    watch_ids = [r[0] for r in conn.execute(
        'SELECT movie_id FROM watchlist WHERE username=?', (username,)).fetchall()]
    conn.close()
    if len(fav_ids) < 3:
        return jsonify({'recommendations': [], 'message': 'Add at least 3 favorites to get recommendations'})
    excluded = set(fav_ids + watch_ids)
    recs = recommender.get_recommendations_for_multiple(fav_ids, top_n=20)
    results = [r for r in recs if r['movie_id'] not in excluded][:8]
    return jsonify({'recommendations': results})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 3000))
    app.run(host='0.0.0.0', port=port, debug=False)
