#!/usr/bin/env python
# encoding: utf-8

__author__ = 'yueyt'
from collections import Counter

import jieba
import jieba.analyse
import numpy as np
from PIL import Image
from wordcloud import WordCloud, STOPWORDS


def get_words(text):
    return jieba.lcut(text)


def get_words_with_stopwords(text, topK=10000):
    jieba.analyse.set_stop_words("data/stop_words.txt")
    tags = jieba.analyse.extract_tags(text, topK=topK)
    return tags


def get_word_count(words, num=20):
    c = Counter(words)
    result = {word: freq for word, freq in c.most_common(num)}
    return result


def get_word_textrank(text, topK=20, withWeight=False):
    tags = jieba.analyse.textrank(text, topK=topK, withWeight=withWeight, allowPOS=('ns', 'n', 'vn', 'v'))
    return tags


def create_word_cloud(word_freq, max_words=300):
    # read the mask image
    # taken from
    # http://www.stencilry.org/stencils/movies/alice%20in%20wonderland/255fk.jpg
    alice_mask = np.array(Image.open('pic/map.jpeg'))

    stopwords = set(STOPWORDS)
    # stopwords.add("said")

    wc = WordCloud(background_color="black", max_words=max_words, mask=alice_mask,
                   stopwords=stopwords, max_font_size=80,
                   random_state=42, font_path='/System/Library/Fonts/PingFang.ttc')
    # generate word cloud
    # wc.generate_from_text(text)
    wc.generate_from_frequencies(word_freq)

    # store to file
    wc.to_file("alice.png")

    # show
    # plt.imshow(wc, interpolation='bilinear')
    # plt.axis("off")
    # plt.figure()
    # plt.imshow(alice_mask, cmap=plt.cm.gray, interpolation='bilinear')
    # plt.axis("off")
    # plt.show()


if __name__ == '__main__':
    with open('data/xi.txt', encoding='utf8') as f:
        text = f.read()
        words = get_words(text)
        most_words = get_word_count(words, num=100)
        print('>>>>', most_words)
        tags = get_word_textrank(text, topK=500, withWeight=True)
        # print(tags)
        tags = {word: freq for word, freq in tags}
        create_word_cloud(tags, max_words=500)
