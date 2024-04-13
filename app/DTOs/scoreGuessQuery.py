from pydantic import BaseModel


class ScoreGuessQuery(BaseModel):
    director_name: str
    movie_title: str

    def __repr__(self) -> str:
        return f"director_name : {self.diretor_name}, movie_title: {self.movie_title}"