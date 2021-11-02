#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/10/22 10:07
# @Author  : fengmq
# @FileName: utils.py
# @Software: PyCharm
import json


def extract():
    with open(r"E:\RecipeQA\data\数据集\train-10-20-2021\crl_srl_10-20-2021.csv", 'r', encoding='utf-8') as f:
        all_data = f.readlines()
    dicts = {}
    question = []
    text = []
    newdoc_id = ''  # 防止报错
    for i in range(len(all_data) - 1):
        item = all_data[i]
        if 'newdoc id' in item:
            if i != 0:
                dicts[newdoc_id] = {}
                dicts[newdoc_id]['question'] = question
                text = " ".join(text)
                dicts[newdoc_id]['text'] = text
            # 清空--
            newdoc_id = item.split("=")[1].strip("\n")
            question = []
            text = []
            continue
        item_next = all_data[i + 1]
        if 'question' in item and 'answer' in item_next:
            qa = item.split("=")[1].strip("\n") + "    " + item_next.split("=")[1].strip("\n")
            question.append(qa)
        if 'text =' in item:
            text.append(item.split("=")[1].strip("\n"))
    fw = open("text_data.json", "w", encoding="utf-8")
    json.dump(dicts, fw, ensure_ascii=False, indent=4)
    fw.close()


def split_to_three():
    f = open("text_data.json", "r", encoding="utf-8")
    data = json.load(f)
    f.close()

    idnum = 1
    idevent = 1
    idother = 1
    dict_num = {}
    dict_event = {}
    dict_other = {}
    for key in data.keys():

        que = data[key]['question']
        text = data[key]['text']
        for i in que:
            q, a = i.split("     ")
            q = q.strip()
            item = {}
            item['key'] = key
            item['question'] = q
            item['answer'] = a
            item['text'] = text
            if q.startswith("How many"):
                dict_num[idnum] = item
                idnum += 1
            elif q.endswith("which comes first?"):
                dict_event[idevent] = item
                idevent += 1
            else:
                dict_other[idother] = item
                idother += 1

    fw = open("number.json", "w", encoding="utf-8")
    json.dump(dict_num, fw, ensure_ascii=False, indent=4)
    fw.close()

    fw = open("event.json", "w", encoding="utf-8")
    json.dump(dict_event, fw, ensure_ascii=False, indent=4)
    fw.close()

    fw = open("other.json", "w", encoding="utf-8")
    json.dump(dict_other, fw, ensure_ascii=False, indent=4)
    fw.close()

def text_of_RecipeQA():
    '''
    得到RecipeQA的text，进行预训练
    :return:
    '''
    fw = open('recipe_corpora.txt','w',encoding='utf-8')
    with open(r"E:\RecipeQA\data\数据集\recipeQA\train.json",'r',encoding='utf-8') as f:
        data = json.load(f)['data']
    for item in data:
        context = item['context'] #list
        for  i in context:
            text=i['body'].strip()
            text=text.strip(" ")
            text = text.replace("  ",' ')
            text = text.replace("\n", ' ')
            if len(text.split(" "))>2:
                fw.write(text+'\n')

    fw.close()
text_of_RecipeQA()