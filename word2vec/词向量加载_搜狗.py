# -*- coding: utf-8 -*-
# @Time: 2023/12/10 0010 18:20
# @Author: Changmeng Yang


from tqdm import  tqdm
import pickle
import numpy as np

if __name__ == "__main__":

    sg_vec = []
    sg_index_2_word = []
    print("加载词向量...")
    with open("sgns.sogou.char/sgns.sogou.char","r",encoding="utf-8") as f:
        all_sg_content = f.read().split("\n")[1:]
        for content in tqdm(all_sg_content):
            s_c = content.strip().split(" ")
            if len(s_c) != 301:
                continue
            sg_index_2_word.append(s_c[0])
            sg_vec.append(s_c[1:])
    sg_word_2_index = {w:i for i,w in enumerate(sg_index_2_word)}
    with open("wordvec.abc","rb") as f:

        w1 = pickle.load(f)

    with open("word2index.123","rb") as f:
        word_2_index = pickle.load(f)
    index_2_word = list(word_2_index)

    print("替换词向量中...")
    for idx,word in enumerate(index_2_word):

        if word not in sg_index_2_word:
            continue
        sg_v = [float(i) for i in   sg_vec[sg_word_2_index[word]]]
        w1[idx] = np.array(sg_v)

    with open("wordvec_new.abc","wb") as f:
        pickle.dump(w1,f)
    print("")
