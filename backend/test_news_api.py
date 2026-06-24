from news_fetcher import search_news

result = search_news(
    "NASA satellite launch"
)

print(result["status"])

for article in result["articles"]:

    print(article["source"]["name"])
    print(article["title"])
    print()