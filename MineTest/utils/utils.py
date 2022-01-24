#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/1/24 13:43
# @Author  : hit-itnlp-fengmq
# @FileName: utils.py
# @Software: PyCharm

import itertools
import json
import os
import re


def useful_tag_8(tags: str):
    '''
    提取第八列的信息。
    example:
        input: "Drop=flour_mixture.4.1.17:eggs.4.1.2:dough.4.1.15|Habitat=medium_oven_-_proof_skillet"
        return :['drop : flour mixture eggs dough', 'habitat : medium oven proof skillet']  ：list

        实际上应该返回
        return :['drop : flour mixture , eggs , dough', 'habitat : medium oven proof skillet']  ：list
    '''
    regex = re.compile('[^a-zA-Z]')  # to remove .1.2.3 and _ in hidden tags
    tags = tags.lower()
    if "|" in tags:
        tags = tags.split("|")
        ans = []
        for tag in tags:
            if tag:
                name, tag = tag.split("=")
                taglist = tag.split(":")
                regextaglist = [regex.sub(' ', _) for _ in taglist]

                alltag = ", ".join(regextaglist)
                ans.append(name + " : " + alltag)
                # replace multiple blankspace with one
                newans = [' '.join(_.split()) for _ in ans]
        if len(newans) != 0:
            return newans
        else:
            return "_"

    else:
        ans = []
        if tags != '_':
            name, tag = tags.split("=")
            taglist = tag.split(":")
            regextaglist = [regex.sub(' ', _) for _ in taglist]

            alltag = ", ".join(regextaglist)
            ans.append(name + " : " + alltag)
            newans = [' '.join(_.split()) for _ in ans]
            if len(newans) != 0:
                return newans
        else:
            return "_"
def part_to_text(partlist: list, text: str):  # change part number to the word
    text_list = text.split()
    for i, _ in enumerate(partlist):
        if _ != '_':
            partlist[i] = text_list[int(_) - 1]
    return partlist


# 提取的数据只有doc_id，文本的tag
def extract_text_and_tags(types: str = "Train", path=r"crl_srl.csv"):
    '''
    提取crl_srl.csv的文本和其对应的tag

    types：Train,Val,Test
    path:crl_srl.csv文件的路径

    输出：
        if types = "Train"
            out_path="train_all_text_tag.json"
        elif types = "Val":
            out_path = "val_all_text_tag.json"
        elif types = "Test":
            out_path = "test_all_text_tag.json"
    '''
    if types not in ["Train", "Val", "Test"]:
        print("参数错误")
        return
    docs_lines = 0
    if types == "Train":
        docs_lines = 219125
    elif types == "Val":
        docs_lines = 29422  # test: 24416 val:29422
    elif types == "Test":
        docs_lines = 24416

    with open(path, 'r', encoding='utf-8') as f:
        all_data = f.readlines()

    all_text_tag = {}  # store all text tag columns
    i = 1
    while i < docs_lines:
        doc_id = all_data[i - 1].split("=")[1].strip("\n")
        text_tags = []
        while i < docs_lines and 'newdoc id' not in all_data[i]:
            item = all_data[i]
            if i + 1 < docs_lines:
                item_next = all_data[i + 1]
            if 'text =' in item:
                item_pre = all_data[i - 1]
                if 'ingredients' in item_pre:
                    i += 1
                    pass
                else:
                    i += 1
                    temp_text_tags = {}
                    texts, tagcols = "", {}

                    tagcols['word'] = []
                    tagcols['lemma'] = []
                    tagcols['pos'] = []
                    tagcols['entity'] = []
                    tagcols['part'] = []
                    tagcols['result'] = []
                    tagcols['hidden'] = []
                    tagcols['coref'] = []
                    tagcols['prdct'] = []
                    tagcols['arg1'] = []
                    tagcols['arg2'] = []
                    tagcols['arg3'] = []
                    tagcols['arg4'] = []
                    tagcols['arg5'] = []
                    tagcols['arg6'] = []
                    tagcols['arg7'] = []
                    tagcols['arg8'] = []
                    tagcols['arg9'] = []
                    tagcols['arg10'] = []

                    while i < docs_lines and all_data[i] and "sent_id" not in all_data[i] and 'newdoc id' not in \
                            all_data[i]:
                        if "newpar id" in all_data[i]:
                            i += 1
                            continue
                        item = all_data[i]
                        tags = item.strip().split("\t")
                        if len(tags) != 1:
                            tagcols['word'].append(tags[1])
                            tagcols['lemma'].append(tags[2])
                            tagcols['pos'].append(tags[3])
                            tagcols['entity'].append(tags[4])
                            tagcols['part'].append(tags[5])
                            tagcols['result'].append(tags[6])
                            # list of hidden tags ['shadow : water', 'habitat : pot']
                            tagcols['hidden'].append(useful_tag_8(tags[7]))
                            tagcols['coref'].append(tags[8])
                            tagcols['prdct'].append(tags[9])
                            tagcols['arg1'].append(tags[10])
                            tagcols['arg2'].append(tags[11])
                            tagcols['arg3'].append(tags[12])
                            tagcols['arg4'].append(tags[13])
                            tagcols['arg5'].append(tags[14])
                            tagcols['arg6'].append(tags[15])
                            tagcols['arg7'].append(tags[16])
                            tagcols['arg8'].append(tags[17])
                            tagcols['arg9'].append(tags[18])
                            tagcols['arg10'].append(tags[19])
                            texts += (" " + tags[1].lower())
                        i += 1
                    temp_text_tags["text"] = texts.strip()
                    tagcols['part'] = part_to_text(tagcols['part'],
                                                   temp_text_tags["text"])  # from coref number to the word
                    temp_text_tags["tagcols"] = tagcols
                    text_tags.append(temp_text_tags)
            else:
                i += 1
        i += 1

        doc_dict = {}
        doc_dict['text_tags'] = text_tags
        all_text_tag[doc_id.strip()] = doc_dict

    if not os.path.exists("data"):
        os.makedirs("data")
    out_path = None
    if types == "Train":
        out_path = "data/train_all_text_tag.json"
    elif types == "Val":
        out_path = "data/val_all_text_tag.json"
    elif types == "Test":
        out_path = "data/test_all_text_tag.json"
    fw = open(out_path, "w", encoding="utf-8")
    json.dump(all_text_tag, fw, ensure_ascii=False, indent=4)
    fw.close()


# 提取QA
def extract_train_val_q_a(types: str = "Train", path=r'E:\RecipeQA\data\数据集\r2vq_train_10_28_2021\train\crl_srl.csv'):
    if types not in ["Train", "Val"]:
        print("参数错误")
        return
    with open(path, 'r', encoding='utf-8') as f:
        all_data = f.readlines()

    docs_lines = 0
    if types == "Train":
        docs_lines = 219125
    elif types == "Val":
        docs_lines = 29422

    list_qa = []  # store all text tag columns
    i = 1
    while i < docs_lines:
        doc_id = all_data[i - 1].split("=")[1].strip("\n")
        while i < docs_lines and 'newdoc id' not in all_data[i]:
            item = all_data[i]
            if i + 1 < docs_lines:
                item_next = all_data[i + 1]
            if 'question' in item and 'answer' in item_next:
                pre, quest = item.split("=")
                ques_id = pre.strip().split(" ")[-1]
                quest = quest.strip().lower()
                answer = item_next.strip().split("=")[-1]

                temp_dict = {}
                temp_dict['q_id'] = ques_id
                temp_dict['question'] = quest
                temp_dict['answer'] = answer
                temp_dict['doc_id'] = doc_id
                list_qa.append(temp_dict)
            i += 1
        i += 1

    if not os.path.exists("data"):
        os.makedirs("data")
    out_path = None
    if types == "Train":
        out_path = r"data/train_qa.json"
    elif types == "Val":
        out_path = r"data/val_qa.json"

    fw = open(out_path, "w", encoding="utf-8")
    json.dump(list_qa, fw, ensure_ascii=False, indent=4)
    fw.close()


def extract_test_q_a(path=r'E:\RecipeQA\data\ITNLP_Semeval2022_Task6\submit\test\crl_srl.csv'):
    '''
    {
        "q_id": "0-1",
        "question": "How many actions does it take to process the minced meat?",
        "doc_id": "f-6VWP66LZ"
    }
    '''
    with open(path, 'r', encoding='utf-8') as f:
        all_data = f.readlines()

    docs_lines = 24416
    list_qa = []  # 存储quest
    i = 1
    while i <= docs_lines:
        doc_id = all_data[i - 1].split("=")[1].strip("\n")  # 获取doc_id

        while i < docs_lines and 'newdoc id' not in all_data[i]:
            item = all_data[i]
            if 'question' in item:
                pre, quest = item.split("=")
                ques_id = pre.strip().split(" ")[-1]
                quest = quest.strip().lower()
                temp_dict = {}
                temp_dict['q_id'] = ques_id
                temp_dict['question'] = quest
                temp_dict['doc_id'] = doc_id
                list_qa.append(temp_dict)
            i += 1
        i += 1

    if not os.path.exists("data"):
        os.makedirs("data")
    fw = open(r"data/test_qa.json", "w", encoding="utf-8")
    json.dump(list_qa, fw, ensure_ascii=False, indent=4)
    fw.close()


def mergeTextTag(columns: tuple, jsonFile_path):
    '''
    merge text and tags from json file in the form:
    text text # TAG = tag # # TAG = tag # text text .

    parameters:
        columns: tuple of str, where the tags are from, tags can from multiple columns
        json: str, the input json file address

    input:
        column idx in tuple
        json data file

    '''
    jsonFile = open(jsonFile_path, 'r', encoding='utf-8')

    file = json.load(jsonFile)
    paddeddata = []

    for doc_id, each_doc in file.items():
        paddeddoc = {}
        paddeddoc['text'] = []
        paddeddoc['doc_id'] = doc_id.strip()

        texttag_list = each_doc['text_tags']  # list

        # allign each word with target columns tags of it
        for _ in texttag_list:
            tags = _['tagcols']
            text_list = _['text'].split(' ')  # the sentence list

            # combine cols of tags together in list.

            if len(columns) == 1:
                tags_to_merge = tags[columns[0]]
                unested = [tags_to_merge]
            else:
                tags_to_merge = [list(t) for t in zip(tags[c] for c in columns)]
                unested = [list(itertools.chain(*sub)) for sub in tags_to_merge]
            # unzip the hidden tag list
            for l in unested:
                for i, v in enumerate(l):
                    if type(v) == list:
                        l[i] = ' # '.join(v)

            # add '#' at the start and end of tags
            newtag = []
            for idx, _ in enumerate(zip(*unested)):  # add tag name and '#' for not meaningful tags
                if not all(l == '_' for l in _):
                    hashpadlist = ['( ' + columns[i] + ' : ' + t + ' ) ' \
                                   for i, t in enumerate(_) if t != '_']
                    hashpad = ' '.join(hashpadlist)
                    newtag.append(hashpad)
                else:
                    newtag.append('')

            paddedtext = ''
            # and merge with the text list

            for text, tag in zip(text_list, newtag):
                paddedtext += text + ' ' + tag
            pt = paddedtext.replace('hidden : ', '')
            # append padded text to the list
            paddeddoc['text'].append(pt)
        paddeddata.append(paddeddoc)
    return paddeddata


def repalce_arg12none():
    # 取出merge之后的" arg1 :"
    data = json.load(open("val.json", 'r', encoding='utf-8'))
    for items in data:
        text = items['text']
        for i in range(len(text)):
            text[i] = text[i].replace(" arg1 :", "")
            text[i] = text[i].replace(" part :", "")
            text[i] = text[i].replace(" arg2 :", "")

    w = open("val2.json", "w", encoding="utf-8")
    json.dump(data, w, ensure_ascii=False, indent=4)
    w.close()


def change_2_dict():
    new_dict = {}
    data = json.load(open("val2.json", 'r', encoding='utf-8'))
    for item in data:
        doc_id = item["doc_id"]
        text = item["text"]
        new_dict[doc_id] = text
    w = open("val3.json", "w", encoding="utf-8")
    json.dump(new_dict, w, ensure_ascii=False, indent=4)
    w.close()


def get_Text_QA(types: str = "Train", path=r"crl_srl.csv"):
    '''
    封装extract_text_and_tags（）
        extract_train_val_q_a（）
        extract_test_q_a（）
    '''
    if types not in ["Train", "Val", "Test"]:
        print("参数错误")
        return
    extract_text_and_tags(types, path)

    if types == "Train":
        extract_train_val_q_a(types, path)
    elif types == "Val":
        extract_train_val_q_a(types, path)
    elif types == "Test":
        extract_test_q_a(path)


def merge_process(types: str = "Train", columns: tuple = ('hidden',), jsonFile_path=r"data/train_all_text_tag.json"):
    if types not in ["Train", "Val", "Test"]:
        print("参数错误")
        return
    padded = mergeTextTag(columns=columns, jsonFile_path=jsonFile_path)

    for items in padded:
        text = items['text']
        for i in range(len(text)):
            text[i] = text[i].replace(" arg1 :", "")
            text[i] = text[i].replace(" part :", "")

    new_dict = {}
    for item in padded:
        doc_id = item["doc_id"]
        text = item["text"]
        new_dict[doc_id] = text

    out_path = None
    if types == "Train":
        out_path = r"train3.json"
    elif types == "Val":
        out_path = r"val3.json"
    elif types == "Test":
        out_path = r"test3.json"
    w = open(out_path, "w", encoding="utf-8")
    json.dump(new_dict, w, ensure_ascii=False, indent=4)
    w.close()


crl_srl_path = {
    "Train": r"E:\RecipeQA\MineTest\datas\r2vq_train_11_16_2021\train\crl_srl.csv",
    "Val": r"E:\RecipeQA\MineTest\datas\Development and Test\r2vq_val_12_03_2021\val\crl_srl.csv",
    "Test": r"E:\RecipeQA\MineTest\datas\Development and Test\r2vq_test_12_03_2021\test\crl_srl.csv"
}

text_tag_file = {
    "Train": r"data/train_all_text_tag.json",
    "Val": r"data/val_all_text_tag.json",
    "Test": r"data/test_all_text_tag.json"
}

# 生成text_tag.json文件和qa.json文件
for key, value in crl_srl_path.items():
    get_Text_QA(key, value)
# merge
for key, value in text_tag_file.items():
    merge_process(key, ('hidden',), value)

