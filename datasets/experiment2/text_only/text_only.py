#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/2/21 22:33
# @Author  : hit-itnlp-fengmq
# @FileName: text_only.py
# @Software: PyCharm
import json

datas = json.load(open(r"train_all_text_tag.json",'r',encoding='utf-8'))

new_dict={}
for doc_id,items in datas.items():
    texts =""
    text_tags = items['text_tags']
    for line in text_tags:
        text = line['text']
        texts+=(" "+text)
    new_dict[doc_id] = texts


w = open('train.json', "w", encoding="utf-8")
json.dump(new_dict, w, ensure_ascii=False, indent=4)
w.close()