# AI-Powered News Verification & Fact Checking System

## Overview

AI-Powered News Verification & Fact Checking System is a web application that helps users verify the authenticity of news articles and claims using:

* Google Fact Check API
* NewsAPI
* Machine Learning Model
* Similarity Matching
* FastAPI Backend
* Streamlit Frontend

## Features

* User Registration & Login
* News Verification Dashboard
* Google Fact Check Integration
* Trusted News Source Verification
* Machine Learning Prediction
* Similarity Score Analysis
* Modern Streamlit UI

## Tech Stack

### Frontend

* Streamlit

### Backend

* FastAPI

### Database

* SQLite

### Machine Learning

* Scikit-Learn
* TF-IDF Vectorization
* Logistic Regression

### APIs

* Google Fact Check API
* NewsAPI

## Project Structure

backend/

* api.py
* verdict_engine.py
* predict.py
* fact_checker.py
* news_fetcher.py

models/

* best_model.jb
* vectorizer.jb

app.py
auth.py

## How to Run

### Install Requirements

pip install -r requirements.txt

### Start FastAPI

uvicorn backend.api:app --reload

### Start Streamlit

streamlit run app.py

## Model Performance

* Dataset Size: 44,898 Articles
* Accuracy: 98.79%

## Author

Uday Kiran Salveru

MCA Final Year Project
