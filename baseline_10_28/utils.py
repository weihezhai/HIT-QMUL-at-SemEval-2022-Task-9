#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/10/22 10:07
# @Author  : fengmq
# @FileName: utils.py
# @Software: PyCharm
import json

import torch
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm


# torch.cuda.set_device(1)


def extract2():
    '''
    从原来的csv提取出问答对json,将原材料单独标注出来
    :return:
    '''
    with open(r"E:\RecipeQA\data\数据集\r2vq_train_10_28_2021\train\crl_srl.csv", 'r', encoding='utf-8') as f:
        all_data = f.readlines()
    dicts = {}
    question = []
    text = []
    meta = []
    ingredients = []
    newdoc_id = ''  # 防止报错
    for i in range(len(all_data) - 1):
        item = all_data[i]
        if 'newdoc id' in item:
            if i != 0:
                dicts[newdoc_id] = {}
                dicts[newdoc_id]['question'] = question
                # text = " ".join(text)
                dicts[newdoc_id]['text'] = text
                dicts[newdoc_id]['meta'] = meta
                dicts[newdoc_id]['ingredients'] = ingredients
            # 清空--
            newdoc_id = item.split("=")[1].strip("\n")
            question = []
            text = []
            meta = []
            ingredients = []
            continue
        item_next = all_data[i + 1]
        if 'question' in item and 'answer' in item_next:
            qa = item.split("=")[1].strip("\n") + "    " + item_next.split("=")[1].strip("\n")
            question.append(qa)
        if 'text =' in item:
            item_pre = all_data[i - 1]
            if 'ingredients' in item_pre:
                ingredients.append(item.split("=")[1].strip("\n"))
            else:
                text.append(item.split("=")[1].strip("\n"))
        if 'metadata:url' in item:
            meta.append(item.split('/')[-1].strip("\n"))
    print(len(dicts))
    fw = open("text_data.json", "w", encoding="utf-8")
    json.dump(dicts, fw, ensure_ascii=False, indent=4)
    fw.close()


def similarity_question_text_SBert():
    '''利用sbert将文本与问题最相似的句子拿出来
    :return:
    '''

    def scores(question: str, text: list, model, top_k: int = 2):
        text_embeddings = model.encode(text, convert_to_tensor=True)
        query_embedding = model.encode(question, convert_to_tensor=True)

        if torch.cuda.is_available():
            text_embeddings.to('cuda')
            query_embedding = torch.unsqueeze(query_embedding, dim=0)
            query_embedding.to('cuda')
            text_embeddings = util.normalize_embeddings(text_embeddings)
            query_embedding = util.normalize_embeddings(query_embedding)
        top_results = util.semantic_search(query_embedding, text_embeddings,
                                           top_k=top_k)  # [[{corpus_id:,score:},{},{}]]
        text_ids = [item['corpus_id'] for item in top_results[0]]
        # 还原顺序
        text_ids.sort()
        result = []
        for id in text_ids:
            result.append(text[id])
        return result

    model = SentenceTransformer('/home/mqfeng/preModels/all-MiniLM-L6-v2')
    f = open(r"/home/mqfeng/R2QA/text_data.json", "r", encoding="utf-8")
    datas = json.load(f)
    f.close()

    dict_new = {}
    for key, value in tqdm(datas.items()):
        meta = value['meta']
        text = value['text']
        ingredients = value['ingredients']
        questions = value['question']

        dict_new[key] = {}
        dict_new[key]['meta'] = meta
        groups = []
        for question in questions:
            group = {}
            ques, ans = question.split("     ")
            result_text = scores(ques, text, model, top_k=2)
            result_ingre = scores(ques, ingredients, model, top_k=2)
            group['question'] = ques
            group['answer'] = ans
            group['text'] = result_ingre + result_text
            groups.append(group)
        dict_new[key]['group'] = groups
        dict_new[key]['ingredients_all'] = ingredients
        dict_new[key]['text_all'] = text
    fw = open("text_data2.json", "w", encoding="utf-8")
    json.dump(dict_new, fw, ensure_ascii=False, indent=4)
    fw.close()


def analyse():
    '''
    all 26503
    first_event 1920
    second_event 1871
    NotAnswer 2790
    num 3306
    :return:
    '''
    all=0
    first_event=0
    second_event = 0
    NotAnswer = 0
    num,how_many=0,0
    match=0
    what,how,where,other=0,0,0,0
    others_all=0
    others_data=[]
    f = open(r"E:\RecipeQA\data\ITNLP_Semeval2022_Task6\old_version\text_data.json", "r", encoding="utf-8")
    datas = json.load(f)
    f.close()

    for key, value in tqdm(datas.items()):
        meta = value['meta']
        text = value['text']
        ingredients = value['ingredients']
        questions = value['question']
        text=(meta[0]+":"+"".join(ingredients)+"."+"".join(text)).lower()
        for question in questions:
            all+=1
            ques, ans = question.split("     ")
            if ques.lower().strip().startswith('how many') and ans !='N/A':
                how_many+=1

            if ans=='the first event':
                first_event+=1
            elif ans=="the second event":
                second_event+=1
            elif ans =='N/A':
                NotAnswer+=1
            elif ans.isdigit():
                num+=1
            elif ans.lower() in text.lower():
                match+=1
            else:
                if ques.lower().strip().startswith('what'):
                    what+=1
                elif ques.lower().strip().startswith('how'):
                    how+=1
                elif ques.lower().strip().startswith('where'):
                    where+=1
                else:
                    other+=1
                others_data.append(question+'\n'+text)
                others_all+=1
    # fw=open("others.txt",'w',encoding='utf-8')
    #
    # for i in others_data:
    #     fw.write(i+'\n\n')
    # fw.close()
    print("all",all)
    print('first_event',first_event)
    print('second_event', second_event)
    print('NotAnswer', NotAnswer)
    print('num', num)
    print('how_many', how_many)
    print('match', match)
    print('what', what)
    print('how', how)
    print('where', where)
    print('other', other)
    print('others_all', others_all)
    pass


def similarity_question_text_common_words():
    def scores(question: str, text: list):
        '''
        :param question: 针对一个问答对
        :param text: 全部的text
        :return:
        # 词干提取与词型还原后续在考虑
        # 大多数文本里的复数在问题仍旧是复数。
        # 主要在于动词： serve-->served, combine-->combining

        # min(3,>0)

        # newdoc id = f-PH5TQR8X  f-PH5TQR8X原始数据集有问题
        '''
        STOP_WORDS = ["What's", "How", "how", 'many']
        score = []
        question_list = question.replace("?", "").lower()  # 去除问题当中的问号
        question_list = question_list.strip().split(" ")
        question_list = [i for i in question_list if i not in stopwords.words('english') + STOP_WORDS]  # 去除没有语义的停用词
        for item in text:
            item = item.replace(".", "").replace(",", "").replace("\"", "").replace(":", "").replace("  ",
                                                                                                     " ").lower()  # 去除句子当中的标点
            item = item.strip().split(" ")
            item = [i for i in item if i not in stopwords.words('english') + STOP_WORDS]  # 去停用词

            inter = set(question_list).intersection(set(item))
            union = set(question_list).union(set(item))
            if len(item) == 0:
                score.append(1)
                print(question)
                print(text)
                print('---------------------------')
            else:
                score.append(len(inter) / len(item))
        result = []
        for i in range(len(text)):
            if score[i] > 0:
                result.append(text[i])
        return result

    f = open("text_data.json", "r", encoding="utf-8")
    datas = json.load(f)
    f.close()

    dict_new = {}
    for key, value in tqdm(datas.items()):

        meta = value['meta']
        text = value['text']
        questions = value['question']

        dict_new[key] = {}
        dict_new[key]['meta'] = meta
        groups = []
        for question in questions:
            group = {}
            ques, ans = question.split("     ")
            result = scores(ques, text)
            group['question'] = ques
            group['answer'] = ans
            group['text'] = ' '.join(result)
            groups.append(group)
        dict_new[key]['group'] = groups
        dict_new[key]['text_all'] = text
    fw = open("text_data2.json", "w", encoding="utf-8")
    json.dump(dict_new, fw, ensure_ascii=False, indent=4)
    fw.close()


def split_to_three():
    '''
    简单将数据集划分为how many ,first-event/second-event,其他类
    :return:
    '''
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
    得到RecipeQA的text，进行增量预训练
    :return:
    '''
    fw = open('recipe_corpora.txt', 'w', encoding='utf-8')
    with open(r"E:\RecipeQA\data\数据集\recipeQA\train.json", 'r', encoding='utf-8') as f:
        data = json.load(f)['data']
    for item in data:
        context = item['context']  # list
        for i in context:
            text = i['body'].strip()
            text = text.strip(" ")
            text = text.replace("  ", ' ')
            text = text.replace("\n", ' ')
            if len(text.split(" ")) > 2:
                fw.write(text + '\n')

    fw.close()


def extract():
    '''
    从原来的csv提取出问答对json,没有标注原材料。详见extract2（）
    :return:
    '''
    with open(r"E:\RecipeQA\data\数据集\r2vq_train_10_28_2021\train\crl_srl.csv", 'r', encoding='utf-8') as f:
        all_data = f.readlines()
    dicts = {}
    question = []
    text = []
    meta = []
    newdoc_id = ''  # 防止报错
    for i in range(len(all_data) - 1):
        item = all_data[i]
        if 'newdoc id' in item:
            if i != 0:
                dicts[newdoc_id] = {}
                dicts[newdoc_id]['question'] = question
                # text = " ".join(text)
                dicts[newdoc_id]['text'] = text
                dicts[newdoc_id]['meta'] = meta
            # 清空--
            newdoc_id = item.split("=")[1].strip("\n")
            question = []
            text = []
            meta = []
            continue
        item_next = all_data[i + 1]
        if 'question' in item and 'answer' in item_next:
            qa = item.split("=")[1].strip("\n") + "    " + item_next.split("=")[1].strip("\n")
            question.append(qa)
        if 'text =' in item:
            text.append(item.split("=")[1].strip("\n"))
        if 'metadata:url' in item:
            meta.append(item.split('/')[-1].strip("\n"))
    print(len(dicts))
    fw = open("text_data.json", "w", encoding="utf-8")
    json.dump(dicts, fw, ensure_ascii=False, indent=4)
    fw.close()
