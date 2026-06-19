from __future__ import annotations

import difflib
import os
import pickle
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template_string, request


BASE_DIR = Path(
    os.environ.get(
        "MOVIE_RECOMMENDER_DATA_DIR",
        "/Users/thanyabegum/Desktop/Intelligence Internship",
    )
)
MODEL_PATH = BASE_DIR / "model.pkl"
SIMILARITY_PATH = BASE_DIR / "similarity.pkl"

app = Flask(__name__)


def load_pickle(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Set MOVIE_RECOMMENDER_DATA_DIR to the folder "
            "that contains model.pkl and similarity.pkl."
        )

    with path.open("rb") as file:
        return pickle.load(file)


def load_recommender_data() -> tuple[pd.DataFrame, object]:
    movies_df = load_pickle(MODEL_PATH)
    similarity_matrix = load_pickle(SIMILARITY_PATH)

    required_columns = {"title"}
    missing = required_columns.difference(movies_df.columns)
    if missing:
        raise ValueError(f"model.pkl is missing required columns: {', '.join(missing)}")

    if len(movies_df) != len(similarity_matrix):
        raise ValueError("model.pkl and similarity.pkl do not contain the same number of movies.")

    return movies_df.reset_index(drop=True), similarity_matrix


movies, similarity = load_recommender_data()
movie_titles = movies["title"].dropna().astype(str).sort_values().tolist()
title_to_index = {title.lower(): index for index, title in movies["title"].astype(str).items()}


def find_movie_title(query: str) -> str | None:
    query = query.strip()
    if not query:
        return None

    exact_match = title_to_index.get(query.lower())
    if exact_match is not None:
        return str(movies.iloc[exact_match]["title"])

    matches = difflib.get_close_matches(query, movie_titles, n=1, cutoff=0.55)
    return matches[0] if matches else None


def recommend_movies(movie_title: str, top_n: int = 10) -> list[dict[str, object]]:
    matched_title = find_movie_title(movie_title)
    if matched_title is None:
        raise ValueError(f"No movie found for '{movie_title}'.")

    movie_index = title_to_index[matched_title.lower()]
    distances = sorted(
        enumerate(similarity[movie_index]),
        reverse=True,
        key=lambda item: item[1],
    )

    recommendations = []
    for index, score in distances[1 : top_n + 1]:
        row = movies.iloc[index]
        recommendations.append(
            {
                "title": str(row["title"]),
                "score": round(float(score), 3),
            }
        )

    return recommendations


PAGE = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Movie Recommender</title>
    <style>
      :root {
        color-scheme: light;
        --ink: #172026;
        --muted: #66717a;
        --line: #d8dee3;
        --paper: #f6f7f8;
        --accent: #d14d35;
        --accent-dark: #9f2f20;
        --panel: #ffffff;
      }

      * {
        box-sizing: border-box;
      }

      body {
        margin: 0;
        min-height: 100vh;
        font-family: Arial, Helvetica, sans-serif;
        color: var(--ink);
        background: var(--paper);
      }

      main {
        width: min(1080px, calc(100% - 32px));
        margin: 0 auto;
        padding: 36px 0;
      }

      .shell {
        display: grid;
        grid-template-columns: minmax(280px, 380px) 1fr;
        gap: 24px;
        align-items: start;
      }

      .panel,
      .result {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
        box-shadow: 0 14px 32px rgba(23, 32, 38, 0.07);
      }

      .panel {
        padding: 22px;
        position: sticky;
        top: 24px;
      }

      h1 {
        margin: 0 0 10px;
        font-size: clamp(2rem, 4vw, 3.6rem);
        line-height: 1;
        letter-spacing: 0;
      }

      .lede {
        color: var(--muted);
        line-height: 1.5;
        margin: 0 0 26px;
      }

      label {
        display: block;
        font-weight: 700;
        margin-bottom: 8px;
      }

      input,
      select,
      button {
        width: 100%;
        min-height: 44px;
        border-radius: 8px;
        font: inherit;
      }

      input,
      select {
        border: 1px solid var(--line);
        padding: 10px 12px;
        background: #fff;
        color: var(--ink);
      }

      .field {
        margin-bottom: 16px;
      }

      button {
        border: 0;
        background: var(--accent);
        color: #fff;
        font-weight: 700;
        cursor: pointer;
      }

      button:hover,
      button:focus {
        background: var(--accent-dark);
      }

      .result {
        overflow: hidden;
      }

      .result header {
        padding: 20px 22px;
        border-bottom: 1px solid var(--line);
      }

      .result h2 {
        margin: 0;
        font-size: 1.25rem;
      }

      .movie-list {
        list-style: none;
        margin: 0;
        padding: 0;
      }

      .movie-list li {
        display: flex;
        justify-content: space-between;
        gap: 16px;
        padding: 15px 22px;
        border-bottom: 1px solid var(--line);
      }

      .movie-list li:last-child {
        border-bottom: 0;
      }

      .score {
        color: var(--muted);
        white-space: nowrap;
      }

      .empty {
        padding: 22px;
        color: var(--muted);
        line-height: 1.5;
      }

      .error {
        margin-top: 16px;
        padding: 12px;
        border-radius: 8px;
        background: #fff2ef;
        color: #872514;
      }

      @media (max-width: 760px) {
        main {
          width: min(100% - 24px, 640px);
          padding: 24px 0;
        }

        .shell {
          grid-template-columns: 1fr;
        }

        .panel {
          position: static;
        }
      }
    </style>
  </head>
  <body>
    <main>
      <h1>Movie Recommender</h1>
      <p class="lede">Pick a movie from the saved TMDB model and get similar recommendations.</p>

      <div class="shell">
        <section class="panel" aria-label="Recommendation form">
          <form method="post" action="/">
            <div class="field">
              <label for="movie">Movie title</label>
              <input id="movie" name="movie" list="movies" value="{{ selected_movie }}" autocomplete="off" required>
              <datalist id="movies">
                {% for title in movie_titles %}
                  <option value="{{ title }}"></option>
                {% endfor %}
              </datalist>
            </div>

            <div class="field">
              <label for="count">Number of recommendations</label>
              <select id="count" name="count">
                {% for option in [5, 10, 15, 20] %}
                  <option value="{{ option }}" {% if option == count %}selected{% endif %}>{{ option }}</option>
                {% endfor %}
              </select>
            </div>

            <button type="submit">Recommend</button>
          </form>

          {% if error %}
            <div class="error">{{ error }}</div>
          {% endif %}
        </section>

        <section class="result" aria-live="polite">
          <header>
            <h2>{{ heading }}</h2>
          </header>
          {% if recommendations %}
            <ol class="movie-list">
              {% for movie in recommendations %}
                <li>
                  <span>{{ movie.title }}</span>
                  <span class="score">{{ movie.score }}</span>
                </li>
              {% endfor %}
            </ol>
          {% else %}
            <div class="empty">Try a title such as The Dark Knight Rises, Avatar, or Spectre.</div>
          {% endif %}
        </section>
      </div>
    </main>
  </body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    selected_movie = request.form.get("movie", "")
    count = int(request.form.get("count", 10))
    recommendations = []
    heading = "Recommendations"
    error = None

    if request.method == "POST":
        try:
            recommendations = recommend_movies(selected_movie, top_n=count)
            matched_title = find_movie_title(selected_movie)
            heading = f"Because you chose {matched_title}"
        except ValueError as exc:
            error = str(exc)

    return render_template_string(
        PAGE,
        movie_titles=movie_titles,
        selected_movie=selected_movie,
        recommendations=recommendations,
        heading=heading,
        count=count,
        error=error,
    )


@app.get("/api/recommend")
def api_recommend():
    movie_title = request.args.get("movie", "")
    count = min(max(request.args.get("count", default=10, type=int), 1), 20)

    try:
        matched_title = find_movie_title(movie_title)
        recommendations = recommend_movies(movie_title, top_n=count)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404

    return jsonify(
        {
            "movie": matched_title,
            "count": len(recommendations),
            "recommendations": recommendations,
        }
    )


@app.get("/api/movies")
def api_movies():
    return jsonify(movie_titles)


if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)
