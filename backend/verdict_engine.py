import re

from predict import predict_news
from news_fetcher import search_news
from fact_checker import check_fact

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_similarity(text1, text2):
    try:

        vectorizer = TfidfVectorizer()

        vectors = vectorizer.fit_transform(
            [text1, text2]
        )

        similarity = cosine_similarity(
            vectors[0],
            vectors[1]
        )[0][0]

        return float(similarity)

    except Exception:

        return 0.0

def verify_news(news_text):

    query = news_text[:200]

    # --------------------------------
    # FACT CHECK SEARCH
    # --------------------------------

    fact_checks = check_fact(query)

    matched_fact_checks = []

    for fact in fact_checks:

        claim_text = fact.get(
            "text",
            ""
        )

        similarity = calculate_similarity(
            news_text,
            claim_text
        )

        if similarity >= 0.50:

            fact["match_score"] = round(
                similarity * 100,
                2
            )

            matched_fact_checks.append(fact)

            matched_fact_checks = sorted(
                matched_fact_checks,
                key=lambda x: x["match_score"],
                reverse=True
)

    # --------------------------------
    # NEWS SEARCH
    # --------------------------------

    news_results = search_news(query)

    articles = news_results.get(
        "articles",
        []
    )

    matched_sources = 0

    article_results = []

    trusted_sources = [

        "BBC News",
        "Reuters",
        "Associated Press",
        "NPR",
        "CNN",
        "The Guardian",
        "Scientific American",
        "The Next Web",
        "ABC News",
        "CBS News"

    ]

    for article in articles:

        title = article.get(
            "title",
            ""
        )

        description = article.get(
            "description",
            ""
        )

        combined_text = (
            title + " " + description
        )

        similarity = calculate_similarity(
            news_text,
            combined_text
        )

        source_name = article.get(
            "source",
            {}
        ).get(
            "name",
            "Unknown"
        )

        if source_name not in trusted_sources:
            continue

        if similarity >= 0.20:
            matched_sources += 1

        article_results.append({

            "source":
                source_name,

            "title":
                title,

            "url":
                article.get(
                    "url",
                    ""
                ),

            "similarity":
                round(
                    similarity * 100,
                    2
                )
        })

    # --------------------------------
    # ML Prediction (Reference Only)
    # --------------------------------

    ml_result = predict_news(news_text)

    prediction = ml_result["prediction"]
    confidence = ml_result["confidence"]

    # Ignore weak ML predictions
    if confidence < 85:
        prediction = "UNCERTAIN"

    # --------------------------------
    # FINAL VERDICT
    # --------------------------------


    matched_fact_checks = sorted(
        matched_fact_checks,
        key=lambda x: x["match_score"],
        reverse=True
    )

    print("=" * 50)

    if matched_fact_checks:

        print(
            "BEST MATCH:",
            matched_fact_checks[0]["text"]
        )

        print(
            "MATCH SCORE:",
            matched_fact_checks[0]["match_score"]
        )

    print("=" * 50)

    if (
        len(matched_fact_checks) > 0
        and
        matched_fact_checks[0]["match_score"] >= 70
    ):

        best_match = matched_fact_checks[0]

        review = best_match.get(
            "claimReview",
            [{}]
        )[0]

        rating = review.get(
            "textualRating",
            ""
        ).lower()

        if "false" in rating:

            verdict = "FALSE CLAIM"

        elif "true" in rating:

            verdict = "TRUE CLAIM"

        else:

            verdict = "FACT CHECK FOUND"

    elif (
        len(matched_fact_checks) > 0
        and
        matched_fact_checks[0]["match_score"] >= 50
    ):

        verdict = "FACT CHECK FOUND"

    elif matched_sources >= 3:

        verdict = "HIGHLY VERIFIED"

    elif matched_sources >= 1:

        verdict = "PARTIALLY VERIFIED"

    else:

        verdict = "NO EVIDENCE FOUND"
    # --------------------------------
    # RESPONSE
    # --------------------------------

    return {

        "prediction":
            prediction,

        "confidence":
            confidence,

        "matched_sources":
            matched_sources,

        "verdict":
            verdict,

        "articles":
            article_results,

        "fact_checks":
            matched_fact_checks
    }

