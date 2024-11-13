import gensim
import re

# Пара слов: досуг, прогулка

# Использование регулярных выражений для извлечения слов в формате "_NOUN"
pat = re.compile("(.*)_NOUN")
# Определение списка позитивных слов
pos = ["моцион_NOUN", "развлечение_NOUN"]
neg = []  # Список негативных слов пустой

word2vec = gensim.models.KeyedVectors.load_word2vec_format("cbow.txt", binary=False)


# Получение 10 наиболее похожих слов на линейную комбинацию
dist = word2vec.most_similar(positive=pos, topn=10)

for i in dist:
    e = pat.match(i[0])
    if e is not None:
        print(e.group(1))
