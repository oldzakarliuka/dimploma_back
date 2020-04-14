import gensim

class WTV:

  def __init__(self, **other):
    self.model = gensim.models.Word2Vec.load('w2v3m.model')

  def closest(self, texts):
    print(texts)
    result = self.model.wv.most_similar(texts)
    return [list(row) for row in result]

  def closest_cosmul(self, pos, neg):
    result = self.model.wv.most_similar_cosmul(positive=pos, negative=neg)
    return [list(row) for row in result]

  def similarity(self, words):
    return self.model.wv.similarity(words[0], words[1]).item()

def prepare_word(word):
  text = word.lower().replace("ё", "е").strip()
  return text


def main():
  model = WTV()
  print(model.closest(['цветок', 'радуга']))
  print(model.closest_cosmul(['радуга'], ['курение']))
  print(model.similarity(('рыба', 'море')))

if __name__ == "__main__":
    main()
