import os
import requests
from dotenv import load_dotenv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

NEWS_API_KEY = os.getenv("NEWS_API_KEY")


def search_news(query):

    url = "https://newsapi.org/v2/everything"

    params = {
        "q": query[:100],
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": 5,
        "apiKey": NEWS_API_KEY
    }

    response = requests.get(
        url,
        params=params
    )

    return response.json()