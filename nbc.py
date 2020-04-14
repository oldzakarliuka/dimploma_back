import joblib
import re
from functools import reduce
class NBC:

    def __init__(self, **other):
        if 'path' in other:
            path = other['path']
        else:
            path = "models/nbc_model.sav"
        self.clf = joblib.load(path)

    def evaluate(self, texts):
        return self.clf.predict(texts)

    def prediction(self, texts):
        predictions = self.evaluate(texts)
        score = reduce(lambda a, b: a+b, predictions) / len(predictions)
        return (score >= .5).item()


def prepare_text(text):
    text = text.lower().replace("ё", "е")
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', text)
    text = re.sub('@[^\s]+', 'USER', text)
    text = re.sub('[^a-zA-Zа-яА-Я1-9.]+', ' ', text)
    text = re.sub('\.$', ' ', text)
    text = re.sub(' +', ' ', text)
    return text.strip()


def main():
    model = NBC()
    print(model.prediction(['Соня, самая милая, умная, красивая киса']))


if __name__ == "__main__":
    main()
