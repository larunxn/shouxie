# -*- coding: utf-8 -*-
# @Time: 2023/12/10 0010 18:20
# @Author: Changmeng Yang

import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def get_most_sim_word(word, topn=1):
    if word not in word_2_index:
        print("词库中没有这个词。")
        return None

    now_vec = w1[word_2_index[word]].reshape(1, -1)

    score = cosine_similarity(now_vec, w1)
    score = score[0]
    sim_word_idx = np.argsort(score)[::-1][:topn]
    sim_word_sco = score[sim_word_idx]
    sim_word = [index_2_word[i] for i in sim_word_idx]

    result = [(i, j) for i, j in zip(sim_word, sim_word_sco)]

    return result

def get_words_sim_score(word1, word2):
    if word1 not in word_2_index or word2 not in word_2_index:
        print("无该词")
        return None

    vec1 = w1[word_2_index[word1]].reshape(1, -1)
    vec2 = w1[word_2_index[word2]].reshape(1, -1)
    score = cosine_similarity(vec1, vec2)
    return score[0][0]


if __name__ == '__main__':
    with open("wordvec.abc","rb") as f:
        w1 = pickle.load(f)

    with open("word2index.123","rb") as f:
        word_2_index = pickle.load(f)
    index_2_word = list(word_2_index)


    while True:
        input_word = input("请输入词语：") # 三角形

        sp_word = input_word.split(" ")
        if len(sp_word) == 1:

            result = get_most_sim_word(input_word,10)
            print(result)
        elif len(sp_word) == 2 :
            result = get_words_score(*sp_word)

            print(result)
