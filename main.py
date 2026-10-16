import os
import pickle
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

#.env, cors configuration
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMG_500 = "https://image.tmdb.org/t/p/w500"

if not TMDB_API_KEY:
    raise RuntimeError("TMDB_API_KEY is missing.Put it in .env as TMDB_API_KEY=e105cc4c4adcc43d998416c958f55d85")

app = FastAPI(
    title="Movie Recommendation API",version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#Path and global vars config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DF_PATH = os.path.join(BASE_DIR, "df.pkl")
INDICES_PATH = os.path.join(BASE_DIR, "indices.pkl")
TFIDF_PATH = os.path.join(BASE_DIR, "tfidf.pkl")
TFIDF_MATRIX_PATH = os.path.join(BASE_DIR, "tfidf_matrix.pkl")

df: Optional[pd.DataFrame] = None
indices_obj: Any = None
tfidf: Any = None
tfidf_matrix: Any = None

TITLE_TO_IDX: Optional[Dict[str, int]] = None

#Models
#Classses are done using Pydantic to validate the data and create models for the API responses.

#MovieCard and MovieDetails models for TMDB API responses
class TMDBMovieCard(BaseModel):
    tmdb_id: int
    title: str
    poster_path: Optional[str] = None
    poster_url: Optional[str] = None
    release_date: Optional[str] = None
    vote_average: Optional[float] = None

class TMDBMovieDetails(BaseModel):
    tmdb_id: int
    title: str
    overview: Optional[str] = None
    poster_path: Optional[str] = None
    release_date: Optional[str] = None
    poster_url: Optional[str] = None
    backdrop_url: Optional[str] = None
    genres: Optional[List[dict]] = None

#To match the response of the recommendation API
class TFIDFRecItem(BaseModel):
    title: str
    score: float
    tmdb: Optional[TMDBMovieCard] = None

#Inheriting the above models to create a response model for the search bundle API
class SearchBundleResponse(BaseModel):
    query: str
    movie_details: TMDBMovieDetails
    tfidf_recommendations: List[TFIDFRecItem]
    genre_recommendations: List[TMDBMovieCard]

#Utility functions

#To read the title of the movie and normalize it for searching in the dataframe
def _norm_title(t: str) -> str:
    """Normalize the title by converting it to lowercase and stripping whitespace."""
    return str(t).strip().lower()

#To create the full image URL from the given path, if the path is None, it returns None
def make_img_url(path: Optional[str]) -> Optional[str]:
    """Construct the full image URL from the given path."""
    if not path:
        return None
    return f"{TMDB_IMG_500}{path}"


async def tmdb_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    q = dict(params)
    q["api_key"] = TMDB_API_KEY

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(f"{TMDB_BASE}{path}", params=q)
    except httpx.RequestError as e:
        raise HTTPException(status_code=500,
        detail=f"TMDB API request error: {type(e).__name__}: {repr(e)}",)
    
    if r.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"TMDB API error: {type(e).__name__}: {repr(e)}",
        )
    return r.json()

#we get the reslts  from the TMDB API and convert them into a list of TMDBMovieCard objects
#limiting the number of results to the specified limit (default is 20) and store in out and return it.
async def tmdb_cards_from_results(
        results: List[Dict[str, Any]], limit: int = 20
) -> List[TMDBMovieCard]:
    out: List[TMDBMovieCard] = []
    for m in (results or [])[:limit]:
        out.append(
            TMDBMovieCard(
                tmdb_id=int(m["id"]),
                title=m.get("title") or m.get("name") or "",
                poster_url=make_img_url(m.get("poster_path")),
                release_date=m.get("release_date"),
                vote_average=m.get("vote_average"),
            )
        )
    
    return out

#Details of a movie from the TMDB API using the movie's TMDB ID. 
#It fetches the data and constructs a TMDBMovieDetails object with the relevant information.
async def tmdb_movie_details(tmdb_id: int) -> TMDBMovieDetails:

    data = await tmdb_get(f"/movie/{tmdb_id}", language="en-US",)
    return TMDBMovieDetails(
        tmdb_id=int(data["id"]),
        title=data.get("title") or "",
        overview=data.get("overview"),
        release_date=data.get("release_date"),
        poster_url=make_img_url(data.get("poster_path")),
        backdrop_url=make_img_url(data.get("backdrop_path")),
        genres=data.get("genres", []) or [],
    )

#Recommendations based on a search query.
#It searches for movies using the TMDB API and returns a list of TMDBMovieCard objects.
async def tmdb_search_movies(query: str, page: int =1) -> Dict[str, Any]:


    return await tmdb_get(
        "/search/movie",
    {   
        "query": query,
        "page": page,
        "language": "en-US",
        "include_adult": False
    },
)

#Gives the first search result for a given query using the TMDB API.
async def tmdb_search_first(query: str) -> Optional[dict]:

    data = await tmdb_search_movies(query = query, page=1)
    results = data.get("results", [])
    return results[0] if results else None

#Converts title to index
def build_title_to_idx_map(indices: Any) -> Dict[str, int]:

    title_to_idx: Dict[str, int] = {}

    if isinstance(indices, dict):
        for k,v in indices.items():
            title_to_idx[_norm_title(k)] = int(v)
    return title_to_idx

    #Pandas series or similar mapping

    try:

        for k, v in indices.items():
            title_to_idx[_norm_title(k)] = int(v)
        return title_to_idx

    except Exception:
         raise RuntimeError
    ("indices.pkl must be dict or pandas Series-like (with .items())")



#Gives local index of a movie title from the TITLE_TO_IDX mapping. If the mapping is not initialized or the title is not found, it raises an HTTPException.
def get_local_idx_by_title(title: str) -> int:
    global TITLE_TO_IDX
    if TITLE_TO_IDX is None:
        raise HTTPException(status_code=500, detail="TF-IDF index map may not initialized")

    key = _norm_title(title)
    if key in TITLE_TO_IDX:
        return int(TITLE_TO_IDX[key])
    raise HTTPException(
        status_code=404, detail=f"Movie title '{title}' not found in local index map"
    )


#Get suitable output from tfidf rectified item 
def tfidf_recommend_titles(
        query_title: str, top_n: int = 10
) -> List[Tuple[str, float]]:
    """Returns list of (title, score) from local df using cosine similarity on TF-IDF matrix.
       Safe against missing columns and rows
    """
    global df, tfidf_matrix

    if df is None or tfidf_matrix is None:
        raise HTTPException(status_code=500, detail="TF-IDF model resources are not loaded")

    try:
        idx = get_local_idx_by_title(query_title)
    except HTTPException as e:
        raise e

    cosine_similarities = np.dot(tfidf_matrix[idx], tfidf_matrix.T).toarray().flatten()
    similar_indices = cosine_similarities.argsort()[::-1][1:top_n + 1]
    similar_scores = cosine_similarities[similar_indices]

    recommendations = [(df.iloc[i]["title"], float(similar_scores[j])) for j, i in enumerate(similar_indices)]
    return recommendations

#Gives Movie posters
async def attach_tmdb_card_by_title(title: str) -> Optional[TMDBMovieCard]:
    """Uses TMDB search by title to fetch poster for a local title.
    if not found returns none (never crashes the endpoint)"""

    try: 
        m = await tmdb_search_first(title)
        if not m:
            return None
        return TMDBMovieCard(
            tmdb_id=int(m["id"]),
            title=m.get("title") or"",
            poster_url=make_img_url(m.get("poster_path")),
            release_date=m.get("release_date"),
            vote_average=m.get("vote_average"),
        )
    except Exception:
        return None

#STARTUP EVENT: Load pickles and initialize global variables when the FastAPI app starts. This function loads the necessary data structures from pickle files, including the DataFrame, indices, TF-IDF matrix, and TF-IDF vectorizer. It also builds a mapping from movie titles to their corresponding indices for efficient lookups during recommendations.
@app.on_event("startup")
def load_pickles():
    global df, indices_obj, tfidf_matrix, tfidf_obj, TITLE_TO_IDX

    #Load df
    with open(DF_PATH, "rb") as f:
        df = pickle.load(f)

    #Load indices
    with open(INDICES_PATH, "rb") as f:
        indices_obj = pickle.load(f)

    #Load tfidf matrix(usually scipy sparse)
    with open(TFIDF_MATRIX_PATH, "rb") as f:
        tfidf_matrix = pickle.load(f)

    #Load tfidf vectorizer (optional, not used directly here)
    with open(TFIDF_PATH, "rb") as f:
        tfidf_obj = pickle.load(f)

    #Build normalized map
    TITLE_TO_IDX = build_title_to_idx_map(indices_obj)

    #sanity
    if df is None or "title" not in df.columns:
        raise RuntimeError("df.pkl must be a pandas DataFrame with a 'title' column")

    
#Routes
#Health Route
@app.get("/health") #Incase any problem arises in API it won't work and we can check the health of the API using this endpoint
def health():
    return {"status": "ok"}
#Home Route
@app.get("/home", response_model=List[TMDBMovieCard]) #This endpoint fetches popular movies from the TMDB API and returns a list of TMDBMovieCard objects.
async def home(
    category: str = Query("popular"),
    limit: int = Query(24, ge=1, le=50),
):
    """Home feed for Streamlit (posters).
       category:
        - trending (trending/move/day)
        - popular, top_rated, upcoming, now_playing (movie/{category})
    """

    try:
        if category == "trending":
            data = await tmdb_get("/trending/movie/day", {"language": "en-US"})
            return await tmdb_cards_from_results(data.get("results", []), limit=limit)

        if category not in {"popular", "top_rated", "upcoming", "now_playing"}:
            raise HTTPException(status_code=400, detail="Invalid category")

        data = await tmdb_get(f"/movie/{category}", {"language": "en-US", "page": 1})
        return await tmdb_cards_from_results(data.get("results", []), limit=limit)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching home feed: {e}")

#Search Route(Gives search results according to the keyword)
@app.get("/tmdb/search")
async def tmdb_search(
    query: str = Query(..., min_length=1),
    page: int = Query(1, ge=1, le=10),
):
    """
    Returns raw TMDB shape with 'results' list.
    Stremlit will use it for:
     -dropdown suggestions
     -grid results 
    """
    return await tmdb_search_movies(query=query, page=page)

#Movie Details
@app.get("/movie/id/{tmdb_id}", response_model=TMDBMovieDetails)
async def movie_details_route(tmdb_id: int):
    return await tmdb_movie_details(tmdb_id)

#Genre Recommendation
@app.get("/recommend/genre", response_model=List[TMDBMovieCard])
async def recommend_genre(
    tmdb_id: int = Query(...),
    limit: int = Query(18, ge=1, le=50)
):
    """
    Given a TMDB Movie ID:
    - Fetch the movie details
    - pick first genre
    - discover movies in that genre(popular)
    """

    details = await tmdb_movie_details(tmdb_id)
    if not details.genres:
        return[]

    genre_id = details.genres[0]["id"]
    discover = await tmdb_get(
        "/discover/movie",
        {
            "with_genres": genre_id,
            "language": "en-US",
            "sort_by": "popularity.desc",
            "page": 1
        },
    )
    cards = await tmdb_cards_from_results(discover.get("results", []), limit=limit)
    return [c for c in cards if c.tmdb_id != tmdb_id]


#TF_IDF ONLY/Recommends titles
@app.get("/recommend/tfidf")
async def recommend_tfidf(
    title: str = Query(..., min_length=1),
    top_n: int = Query(10, ge=1, le=50),
):
    recs = tfidf_recommend_titles(title, top_n=top_n)
    return [{"title": t, "score": s} for t, s in recs]

@app.get("/movie/search", response_model=SearchBundleResponse)
async def search_bundle(
    query: str = Query(..., min_length=1),
    tfidf_top_n: int = Query(12, ge=1, le=30),
    genre_limit: int = Query(12, ge=1, le=30),
):
    """
    This endpoint is for when you have a selected movie and want:
    -movie details
    -TF-IDF recommendations(local)+posters
    -Genre recommendations (TMDB)+posters
    """
    best = await tmdb_search_first(query)
    if not best:
        raise HTTPException(
            status_code=404, detail=f"No TMDB movie found for query: {query}"
        )

    tmdb_id = int(best["id"])
    details = await tmdb_movie_details(tmdb_id)

    # 1) TF-IDF recommendations (never crash endpoint)
    tfidf_items: List[TFIDFRecItem] = []

    recs: List[Tuple[str, float]] = []
    try:
        # try local dataset by TMDB title
        recs = tfidf_recommend_titles(details.title, top_n=tfidf_top_n)
    except Exception:
        # fallback to user query
        try:
            recs = tfidf_recommend_titles(query, top_n=tfidf_top_n)
        except Exception:
            recs = []

    for title, score in recs:
        card = await attach_tmdb_card_by_title(title)
        tfidf_items.append(TFIDFRecItem(title=title, score=score, tmdb=card))

    # 2) Genre recommendations (TMDB discover by first genre)
    genre_recs: List[TMDBMovieCard] = []
    if details.genres:
        genre_id = details.genres[0]["id"]
        discover = await tmdb_get(
            "/discover/movie",
            {
                "with_genres": genre_id,
                "language": "en-US",
                "sort_by": "popularity.desc",
                "page": 1,
            },
        )
        cards = await tmdb_cards_from_results(
            discover.get("results", []), limit=genre_limit
        )
        genre_recs = [c for c in cards if c.tmdb_id != details.tmdb_id]

    return SearchBundleResponse(
        query=query,
        movie_details=details,
        tfidf_recommendations=tfidf_items,
        genre_recommendations=genre_recs,
    )
























