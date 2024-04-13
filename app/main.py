from typing import Union
from routes.imdb import score_router
from fastapi import FastAPI
import os

os.environ["MODEL"] = "rf"

app = FastAPI()

app.include_router(score_router)

@app.get("/")
def read_root():
    return {"Hello": "World"}

