#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/1/4 14:21
# @Author  : hit-itnlp-fengmq
# @FileName: utils.py
# @Software: PyCharm
import json
data = json.load(open("train_qa.json",'r',encoding='utf-8'))
new = []
for items in data:
    if items["answer"]==' N/A':
        continue
    new.append(items)
w = open("train_qa2.json", "w", encoding="utf-8")
json.dump(new, w, ensure_ascii=False, indent=4)
w.close()