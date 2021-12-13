#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/12/12 13:54
# @Author  : hit-itnlp-fengmq
# @FileName: utils.py
# @Software: PyCharm
import itertools
import json
import re


def useful_tag_8(tags: str):
    '''
    提取第八列的信息。
    example:
        input: "Drop=flour_mixture.4.1.17:eggs.4.1.2:dough.4.1.15|Habitat=medium_oven_-_proof_skillet"
        return :['drop : flour mixture eggs dough', 'habitat : medium oven proof skillet']  ：list
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

                alltag = " ".join(regextaglist)
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

            alltag = " ".join(regextaglist)
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
def extract_text_and_tags(Train=True, path=r"crl_srl.csv"):
    '''
    提取crl_srl.csv的文本和其对应的tag

    Train:是否为训练集
    path:crl_srl.csv文件的路径

    输出：
        if Train:
            out_path="train_all_text_tag.json"
        else:
            out_path = "test_all_text_tag.json"
    '''
    if Train:
        docs_lines = 219124
    else:
        docs_lines = 24416
    with open(path, 'r', encoding='utf-8') as f:
        all_data = f.readlines()

    all_text_tag = []  # store all text tag columns
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
        doc_dict['doc_id'] = doc_id
        doc_dict['text_tags'] = text_tags
        all_text_tag.append(doc_dict)
    if Train:
        out_path = "train_all_text_tag.json"
    else:
        out_path = "test_all_text_tag.json"
    fw = open(out_path, "w", encoding="utf-8")
    json.dump(all_text_tag, fw, ensure_ascii=False, indent=4)
    fw.close()


# 训练集提取只有doc_id，question, answer
def extract_train_q_a(path=r'E:\RecipeQA\data\数据集\r2vq_train_10_28_2021\train\crl_srl.csv'):
    with open(path, 'r', encoding='utf-8') as f:
        all_data = f.readlines()

    docs_lines = 219124
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
    fw = open(r"train\train_qa.json", "w", encoding="utf-8")
    json.dump(list_qa, fw, ensure_ascii=False, indent=4)
    fw.close()


def extract_test_q(path=r'E:\RecipeQA\data\ITNLP_Semeval2022_Task6\submit\test\crl_srl.csv'):
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
    fw = open(r"test\test_q.json", "w", encoding="utf-8")
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

    for each_doc in file:
        paddeddoc = {}
        paddeddoc['text'] = []
        paddeddoc['doc_id'] = each_doc['doc_id'].strip()

        texttag_list = each_doc['text_tags']  # list

        # allign each word with target columns tags of it
        for _ in texttag_list:
            tags = _['tagcols']
            text_list = _['text'].split(' ')  # the sentence list

            # combine cols of tags together in list.
            tags_to_merge = [list(t) for t in zip(tags[c] for c in columns)]
            unested = [list(itertools.chain(*sub)) for sub in tags_to_merge]

            # unzip the hidden tag list
            for l in unested:
                for i, v in enumerate(l):
                    if type(v) == list:
                        l[i] = ', '.join(v)

            # add '#' at the start and end of tags
            newtag = []
            for idx, _ in enumerate(zip(*unested)):  # add tag name and '#' for not meaningful tags
                if not all(l == '_' for l in _):
                    hashpadlist = ['# ' + columns[i] + ' : ' + t + ' # ' \
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

# path =r"test/test_all_text_tag.json"
# padded = mergeTextTag(columns=('arg9','arg10'), jsonFile_path=path)
# w = open("text4.json", "w", encoding="utf-8")
# json.dump(padded, w, ensure_ascii=False, indent=4)
# w.close()
import  os
s_path=r"/home/mqfeng/code/RecipeQA/EM/save"
sub_path=os.path.join(s_path,"model"+str(0))
os.mkdir(sub_path)
