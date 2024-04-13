
from model.dataframe import IMDB_df
from fastapi.responses import HTMLResponse, JSONResponse
from model.rf import get_model
from fastapi import FastAPI, APIRouter, Request
from DTOs.scoreGuessQuery import ScoreGuessQuery
from fastapi.templating import Jinja2Templates
import pandas as pd


score_router = APIRouter(prefix="/score", tags=["Score"])
templates = Jinja2Templates(directory="templates")


@score_router.post("/guess")
def get_score(score: ScoreGuessQuery):
    print(score.director_name)
    print(score.movie_title)
    df = IMDB_df.get_df()
    df.str_to_num
    temp_dict_director = {value: key for key, value in df.num_to_str["director_name"].items()}
    temp_dict_movie = {value: key for key, value in df.num_to_str["movie_title"].items()}
    director = temp_dict_director[score.director_name]
    movie = temp_dict_movie[score.movie_title]
    print(director)
    print(movie)
    
    df = IMDB_df.get_df()
    model = get_model(df)
    row = df[(df["director_name"] == director) & (df["movie_title"] == movie)]
    print(f"score : {score}")
    if row.size > 0:
        row = IMDB_df(row.iloc[0]).T
        real_score = row["imdb_score"]
        print("real score", real_score)
        real_score_str = df.num_to_str["imdb_score"][int(real_score.iloc[0])]
        row.drop("imdb_score", axis=1,  inplace=True)
   
    
        result = model.predict(row)[0]
        str_result = df.num_to_str["imdb_score"][result]


        return JSONResponse([real_score_str, str_result])
    

    return "no result"


@score_router.post("/data")
def columns():
    df = IMDB_df.load_imdb().dropna(subset=["director_name", "movie_title"])
    return JSONResponse({"directors": list(df["director_name"].str.strip()), "movies": list(df["movie_title"].str.strip())})


@score_router.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    df = IMDB_df.load_imdb().dropna(subset=["director_name", "movie_title"])
    return templates.TemplateResponse("index.html", {"request": request})
    