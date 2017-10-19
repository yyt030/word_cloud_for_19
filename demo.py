#!/usr/bin/env python
# encoding: utf-8

__author__ = 'yueyt'
from collections import Counter, OrderedDict

import jieba
import jieba.analyse
import numpy as np
from PIL import Image
from wordcloud import WordCloud, STOPWORDS


def without_stopwords(words):
    with open('data/stop_words.txt', encoding='utf8') as f:
        stop_words = f.read()
        for k in stop_words:
            words.pop(k, None)

    words.pop(' ', None)
    return words


def get_words_without_stopwords(text, topK=20):
    jieba.analyse.set_idf_path("data/SogouLabDic.dic")
    tags = jieba.analyse.extract_tags(text, topK=topK, withWeight=True, allowPOS=())
    return dict(tags)


def get_word_textrank(text, topK=20):
    #tags = jieba.analyse.textrank(text, topK=topK, withWeight=True, allowPOS=('ns', 'n', 'vn', 'v'))
    tags = jieba.analyse.textrank(text, topK=topK, withWeight=True, allowPOS=('ns','n'))
    return dict(tags)


def create_word_cloud(word_freq, to_file='alice.png', max_words=300):
    alice_mask = np.array(Image.open('pic/bank.png'))

    stopwords = set(STOPWORDS)
    # stopwords.add("said")

    wc = WordCloud(background_color="white", max_words=max_words, mask=alice_mask,
                   stopwords=stopwords, max_font_size=80, random_state=42,
                   font_path='/System/Library/Fonts/PingFang.ttc')
    # generate word cloud
    # wc.generate_from_text(text)
    wc.generate_from_frequencies(word_freq)

    # store to file
    wc.to_file(to_file)


if __name__ == '__main__':
    max_num = 300
    with open('data/xi.txt', encoding='utf8') as f:
        text = f.read()
        words = jieba.lcut(text)
        most_words = Counter(words).most_common(max_num)
        most_words = without_stopwords(OrderedDict(most_words))
        print('>>>>', most_words)
        create_word_cloud(most_words, to_file='common.png', max_words=max_num)
        # idf
        most_words2 = get_words_without_stopwords(text)
        print('>>> TF-IDF  ', most_words2)
        create_word_cloud(most_words2, to_file='tf-idf.png', max_words=max_num)
        # text rank
        most_words3 = get_word_textrank(text, topK=max_num)
        print('>>> textRank', most_words3)
        create_word_cloud(most_words3, to_file='bank-n2.png', max_words=max_num)
