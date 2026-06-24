# test_dataset.py

from predict import predict_news

tests = [
    "Reuters reported the Federal Reserve kept interest rates unchanged.",
    "NASA announced a successful satellite launch.",
    "The Earth is flat.",
    "COVID vaccine causes infertility in all women.",
    "Scientists discovered water on Mars.",
]

for t in tests:
    print(t)
    print(predict_news(t))
    print("-"*50)