import gensim
import re

# Использование регулярных выражений для извлечения слов в формате "_NOUN"
pat = re.compile("(.*)_NOUN")
# Определение списка позитивных слов
pos = ["досуг_NOUN", "прогулка_PROPN"]
neg = []  # Список негативных слов пустой

word2vec = gensim.models.KeyedVectors.load_word2vec_format("cbow.txt", binary=False)


# Получение 10 наиболее похожих слов на линейную комбинацию
dist = word2vec.most_similar(positive=pos, topn=10)


# Вывод слов, которые соответствуют формату "_NOUN"
for i in dist:
    e = pat.match(i[0])
    if e is not None:
        print(e.group(1))