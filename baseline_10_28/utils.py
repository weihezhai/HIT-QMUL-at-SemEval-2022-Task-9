#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/10/22 10:07
# @Author  : fengmq
# @FileName: utils.py
# @Software: PyCharm
import json
import random

import numpy as np
import torch
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
    top3 :match有6400条
    all:match 7916条
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


def N_A():
    '''阈值的设置及对于准确率
    0.51 0.973
    0.52 0.977
    0.53 0.981
    0.54 0.986
    0.55 0.987
    0.56 0.988
    0.57 0.9906
    0.58 0.9928
    '''  #

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
        top_results = [item['score'] for item in top_results[0]]

        return np.mean(top_results)

    Sbert = SentenceTransformer(r'/home/mqfeng/preModels/all-MiniLM-L6-v2')
    path = 'text_data.json'
    f = open(path, 'r', encoding='utf-8')
    json_data = json.load(f)
    f.close()
    for score_temp in [0.55, 0.56, 0.57, 0.58]:
        num_all = 0
        correct = 0
        for key, value in tqdm(json_data.items()):
            questions = value['question']
            text = value['text']
            for question in questions:
                ques, answer = question.split("     ")
                if answer == r'N/A':
                    num_all += 1
                    score = scores(question, text, Sbert)
                    if score < score_temp:
                        correct += 1
        print(correct / num_all)


def analyse():
    '''
    all 26503
    first_event 1920
    second_event 1871
    NotAnswer 2790
    num 3306
    :return:
    '''
    all = 0
    first_event = 0
    second_event = 0
    NotAnswer = 0
    num, how_many = 0, 0
    match = 0
    what, how, where, other = 0, 0, 0, 0
    others_all = 0

    event_data = []
    NotAnswer_data = []
    num_data = []
    match_data = []
    what_data = []
    how_data = []
    where_data = []
    other_data = []
    others_data = []
    f = open(r"E:\RecipeQA\data\ITNLP_Semeval2022_Task6\old_version\text_data.json", "r", encoding="utf-8")
    datas = json.load(f)
    f.close()

    for key, value in tqdm(datas.items()):
        meta = value['meta']
        text = value['text']
        ingredients = value['ingredients']
        questions = value['question']
        text = (meta[0] + ":" + "".join(ingredients) + "." + "".join(text)).lower()
        for question in questions:
            all += 1
            ques, ans = question.split("     ")
            if ques.lower().strip().startswith('how many') and ans != 'N/A':
                how_many += 1

            if ans == 'the first event':
                first_event += 1
                event_data.append(meta[0] + '     ' + question + '\n' + text)
            elif ans == "the second event":
                second_event += 1
                event_data.append(meta[0] + '     ' + question + '\n' + text)
            elif ans == 'N/A':
                NotAnswer += 1
                NotAnswer_data.append(meta[0] + '     ' + question + '\n' + text)
            elif ans.isdigit():
                num_data.append(meta[0] + '     ' + question + '\n' + text)
                num += 1
            elif ans.lower() in text.lower():
                match_data.append(meta[0] + '     ' + question + '\n' + text)
                match += 1
            else:
                if ques.lower().strip().startswith('what'):
                    what_data.append(meta[0] + '     ' + question + '\n' + text)
                    what += 1
                elif ques.lower().strip().startswith('how'):
                    how_data.append(meta[0] + '     ' + question + '\n' + text)
                    how += 1
                elif ques.lower().strip().startswith('where'):
                    where_data.append(meta[0] + '     ' + question + '\n' + text)
                    where += 1
                else:
                    other_data.append(meta[0] + '     ' + question + '\n' + text)
                    other += 1
                others_data.append(meta[0] + '     ' + question + '\n' + text)
                others_all += 1
    with open(r"E:\RecipeQA\data\subdatas\others.txt", 'w', encoding='utf-8') as f:
        for i in others_data:
            f.write(i + '\n\n')
    with open(r"E:\RecipeQA\data\subdatas\other.txt", 'w', encoding='utf-8') as f:
        for i in other_data:
            f.write(i + '\n\n')
    with open(r"E:\RecipeQA\data\subdatas\where.txt", 'w', encoding='utf-8') as f:
        for i in where_data:
            f.write(i + '\n\n')
    with open(r"E:\RecipeQA\data\subdatas\how.txt", 'w', encoding='utf-8') as f:
        for i in how_data:
            f.write(i + '\n\n')
    with open(r"E:\RecipeQA\data\subdatas\what.txt", 'w', encoding='utf-8') as f:
        for i in what_data:
            f.write(i + '\n\n')
    with open(r"E:\RecipeQA\data\subdatas\match.txt", 'w', encoding='utf-8') as f:
        for i in match_data:
            f.write(i + '\n\n')
    with open(r"E:\RecipeQA\data\subdatas\num.txt", 'w', encoding='utf-8') as f:
        for i in num_data:
            f.write(i + '\n\n')
    with open(r"E:\RecipeQA\data\subdatas\NotAnswer.txt", 'w', encoding='utf-8') as f:
        for i in NotAnswer_data:
            f.write(i + '\n\n')
    with open(r"E:\RecipeQA\data\subdatas\event.txt", 'w', encoding='utf-8') as f:
        for i in event_data:
            f.write(i + '\n\n')
    print("all", all)
    print('first_event', first_event)
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









def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


setup_seed(2021)

def scores(question: str, text: list, model, top_k: int = 4):
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

def split_data(Sbert, path=r'text_data.json', fold=0):
    '''
    :param path: text_data.json的路径
    :param fold: 第几折，默认总共五折交叉
    :return: 返回划分好的训练集与验证集,是一个list，list的元素为一个字典，包含{qa：''，text：''}，其中qa以四个空格分割，q,a=qa.split("     ")
    '''
    f = open(path, 'r', encoding='utf-8')
    json_data = json.load(f)
    f.close()

    datas = []
    for key, value in json_data.items():
        meta = value['meta']
        questions = value['question']
        ingredients = value['ingredients']
        text = value['text']
        text_str = (meta[0] + ":" + "".join(ingredients) + "." + "".join(text)).lower()
        for question in questions:
            ques, answer = question.split("     ")

            if answer.lower() in text_str.lower() and not answer.isdigit():
                data = {}
                data['question'] = ques
                data['answer'] = answer
                # data['text'] = [meta, ingredients, text]
                data['text'] = text
                datas.append(data)
    num_data = len(datas)
    num_test = num_data // 5
    random.shuffle(datas)
    test = datas[fold * num_test:(fold + 1) * num_test]
    if fold == 0:
        train = datas[num_test:]
    else:
        train = datas[:num_test * fold]
        train.extend(datas[num_test * (fold + 1):])
    # 处理验证集加速Sbert
    print(len(test))
    # new_test = []
    # for i in tqdm(test):
    #     meta, ingredients, text = i['text']
    #     question = i['question']
    #     answer = i['answer']
    #     result_text = scores(question, text, Sbert, top_k=4)
    #     result_ingres = scores(question, ingredients, Sbert, top_k=2)
    #     result = meta[0] + "".join(result_ingres + result_text)
    #     result1 = (result_ingres + result_text)
    #     if answer.lower() in result.lower():
    #         data = {}
    #         data['question'] = question
    #         data['answer'] = answer.lower()
    #         data['text'] = result1
    #         new_test.append(data)
    fw = open("test_all.json", "w", encoding="utf-8")
    json.dump(test, fw, ensure_ascii=False, indent=4)
    fw.close()
    # print(len(new_test))
Sbert = SentenceTransformer(r'D:\Anaconda\learn\_Bert\pre_train_model\all-MiniLM-L6-v2')
split_data(Sbert)