# 1. load data
# 2. get word bag
# 3. get tf-idf
# 4. similarity
import os
import codecs
from gensim import corpora, models, similarities
from collections import Counter
from itertools import chain
import heapq
import numpy as np
import pandas as pd
from timeit import default_timer as timer


def nv_filter(property):
    return property in {'ns', 'n', 'vn', 'v'}


def stopwords_filter(property):
    return property not in {'w', 'y', 'c', 'u'}


def load(property_filter=None, min_freq=2):
    if property_filter is None:
        property_filter = stopwords_filter

    ids = []
    docs = []

    # get id, docs and do basic filter
    with codecs.open('data.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            if len(line) == 0 or line[0] != '1': continue

            items = line.strip().split()
            id, p = items[0].rsplit('-', 1)
            # filter stopwords
            words = [item.rsplit('/', 1)[0] for item in items[1:] if property_filter(item.rsplit('/', 1)[1])]

            if p == '001/m':
                ids.append(id)
                docs.append(words)
            else:
                docs[-1].extend(words)

    # remove infrequent words
    word_cnt = Counter(chain(*docs))
    docs = [[word for word in doc if word_cnt[word] >= min_freq] for doc in docs]
    return ids, docs


def gensim_model(filename, docs, model='vsm'):
    dic = corpora.Dictionary(docs)
    corpus = [dic.doc2bow(text) for text in docs]
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    if model == 'lsi':
        lsi = models.LsiModel(corpus_tfidf, num_topics=8)
        corpus_tfidf = lsi[corpus_tfidf]

    index = similarities.MatrixSimilarity(corpus_tfidf)
    with codecs.open('%s_%s.csv' % (filename, model), 'w', encoding='utf-8') as f:
        for sims in index[corpus_tfidf]:
            f.write(','.join(map(str, sims)) + '\n')


def load_mx(filename):
    return pd.read_csv(filename, header=None).values


def save_models(name, filter):
    ids, docs = load(filter)
    t1 = timer()
    gensim_model(name, docs)
    t2 = timer()
    gensim_model(name, docs, 'lsi')
    t3 = timer()
    print('vsm: %d seconds' % (t2 - t1))
    print('lsi: %d seconds' % (t3 - t2))


def similarContrast():
    if not os.path.exists('nv_vsm.csv'):
        print('Use only /n /ns /v /vn to build model and calculate similarity')
        save_models('nv', nv_filter)

    if not os.path.exists('stop_vsm.csv'):
        print('Filter stop words to build model and calculate similarity')
        save_models('stop', stopwords_filter)

    nv_vsm_similar = load_mx('nv_lsi.csv')
    stop_vsm_similar = load_mx('stop_lsi.csv')
    mi = abs(nv_vsm_similar - stop_vsm_similar)

    ac01 = len(np.where(mi > 0.1)[0])
    ac02 = len(np.where(mi > 0.2)[0])
    all_len = mi.size
    print('error > 0.1:', ac01 / all_len)
    print('error > 0.2:', ac02 / all_len)


def top10(filename):
    ids, _ = load()
    ids = np.array(ids)
    simiMx = load_mx(filename)
    with open('top10' + filename, 'w') as f:
        for i, line in enumerate(simiMx):
            # top = np.argsort(-line)
            top = np.argwhere(line > 0.8).flatten()
            if len(top) > 1:
                f.write(ids[i] + ":" + ','.join(list(ids[top])) + '\n')


if __name__ == '__main__':
    top10('nv_vsm.csv')
    top10('stop_vsm.csv')
    top10('nv_lsi.csv')
    top10('stop_lsi.csv')
