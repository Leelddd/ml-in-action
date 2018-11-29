# !/usr/bin/python
# -*- coding:utf-8 -*-

import codecs
import jieba
import jieba.posseg


def cut():
    with codecs.open('data/26.novel.txt', encoding='utf-8') as f:
        str = f.read()

    seg = jieba.posseg.cut(str)
    for s in seg:
        print(s.word, s.flag, '|')


if __name__ == "__main__":
    cut()
