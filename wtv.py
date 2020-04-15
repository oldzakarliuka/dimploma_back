import gensim

class WTV:

  def __init__(self, **other):
    self.model = gensim.models.Word2Vec.load('w2v3m.model')

  def closest(self, **kwords):
    if 'words' in kwords:
      result = self.model.wv.most_similar(kwords['words'])
    elif 'pos' in kwords and 'neg' in kwords:
      result = self.model.wv.most_similar(positive=kwords['pos'], negative=kwords['neg'])
    elif 'pos' in kwords:
      result = self.model.wv.most_similar(positive=kwords['pos'])
    else:
      result = self.model.wv.most_similar(negative=kwords['neg'])

    return [list(row) for row in result]

  def closest_cosmul(self, **kwords):
    if 'words' in kwords:
      result = self.model.wv.most_similar(kwords['words'])
    elif 'pos' in kwords and 'neg' in kwords:
      result = self.model.wv.most_similar_cosmul(positive=kwords['pos'], negative=kwords['neg'])
    elif 'pos' in kwords:
      result = self.model.wv.most_similar_cosmul(positive=kwords['pos'])
    else:
      result = self.model.wv.most_similar_cosmul(negative=kwords['neg'])

    return [list(row) for row in result]

  def similarity(self, words):
    return self.model.wv.similarity(words[0], words[1]).item()

def prepare_word(word):
  text = word.lower().replace("ё", "е").strip()
  return text


def main():
  model = WTV()
  print(model.closest(words=['цветок']))
  print(model.closest_cosmul(words=['радуга']))
  print(model.similarity(('рыба', 'море')))

if __name__ == "__main__":
    main()
