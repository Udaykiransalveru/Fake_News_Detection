import os
import requests
from dotenv import load_dotenv

load_dotenv()

FACT_CHECK_API_KEY = os.getenv("FACT_CHECK_API_KEY")


def check_fact(query):

    try:

        url = (
            "https://factchecktools.googleapis.com/v1alpha1/claims:search"
        )

        params = {
            "query": query,
            "key": FACT_CHECK_API_KEY
        }

        response = requests.get(
            url,
            params=params,
            timeout=10
        )

        data = response.json()

        return data.get(
            "claims",
            []
        )

    except Exception as e:

        print(
            f"Fact Check Error: {e}"
        )

        return []