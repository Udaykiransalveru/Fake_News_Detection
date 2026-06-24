from fastapi import FastAPI
from pydantic import BaseModel

from verdict_engine import verify_news

app = FastAPI(
    title="Fake News Detection API",
    version="1.0"
)


class NewsRequest(BaseModel):
    news_text: str


@app.get("/")
def home():

    return {
        "message": "Fake News Detection API Running"
    }


@app.post("/verify")
def verify(request: NewsRequest):

    result = verify_news(
        request.news_text
    )

    return result