import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, load_model
import joblib
import tensorflow as tf


class RNN:
  
  def __init__(self, **other):
        if 'path' in other:
            path = other['path']
        else:
            path = "models/rnn.h5"
        self.clf = load_model(path)
        
        if 'seq_len' in other:
          self.MAX_SEQUENCE_LENGTH = other['seq_len']
        else:
          self.MAX_SEQUENCE_LENGTH = 80
        
        self.tokenizer = joblib.load('./models/tokenizer.sav')
            
  def evaluate(self, texts):

    seq = self.tokenizer.texts_to_sequences(texts)
    ps = pad_sequences(seq, maxlen=self.MAX_SEQUENCE_LENGTH)
    return self.clf.predict(ps).tolist()
  
  def prediction(self, texts):
    predictions = self.evaluate(texts)
    sum = 0
    for elem in predictions:
      if elem[0] > 0.5:
        sum += 1

    score = sum / len(predictions)
    return score >= .5


def main():
    model = RNN()
    print(type(model.evaluate(['пользуюсь уже полтора года так что удалось уже подробно протестировать я очень доволен', 'цифры которые приведу в посте актуальны для конкретной квартиры'])))
    print(model.evaluate(['пользуюсь уже полтора года так что удалось уже подробно протестировать я очень доволен', 'цифры которые приведу в посте актуальны для конкретной квартиры']))
    print(model.prediction(['пользуюсь уже полтора года так что удалось уже подробно протестировать я очень доволен', 'цифры которые приведу в посте актуальны для конкретной квартиры']))

if __name__ == "__main__":
    main()
