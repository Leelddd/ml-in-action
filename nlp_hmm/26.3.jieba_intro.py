# !/usr/bin/python
# -*- coding:utf-8 -*-

import codecs
import jieba
import jieba.posseg


def cut_and_save(file_path, write_to):
    with codecs.open(file_path, encoding='utf-8') as f:
        str = f.read()

    seg = jieba.posseg.cut(str)

    word_seq = [s.word for s in seg]
    with codecs.open(write_to, 'w', encoding='utf-8') as f:
        f.write(' '.join(word_seq))


if __name__ == "__main__":
    train_file_path = 'data/train'
    train_write_to = 'data/train_cut.txt'
    test_file_path = 'data/test'
    test_write_to = 'data/test_cut.txt'
    cut_and_save(train_file_path, train_write_to)
    cut_and_save(test_file_path, test_write_to)
