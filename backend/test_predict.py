# test_predict.py

from predict import predict_news

news = """
NASA successfully launched a new communication satellite from Cape Canaveral.
The mission was completed on Tuesday and the satellite will improve internet
connectivity in remote regions around the world.
"""

print(predict_news(news))

