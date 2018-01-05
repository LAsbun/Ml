#!/usr/bin/env python3
# encoding: utf-8

"""
@author: sws
@software: PyCharm
@file: ciyun.py
@time: 12/1/17 3:57 PM
@desc:
"""
import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import jieba

font=os.path.join(os.path.dirname(__file__), "DroidSansFallbackFull.ttf")
text = open('/home/sws/ML/ciyun.txt').read()

wordlist_after_jieba = jieba.cut(text, cut_all = True)
# print(list(wordlist_after_jieba))
wl_space_split = " ".join(wordlist_after_jieba)

wordcloud = WordCloud(font_path=font).generate(text)
wordcloud = WordCloud(font_path=font,max_font_size=40).generate(text)
print(wordcloud)
plt.imshow(wordcloud)
plt.axis("off")
plt.show()