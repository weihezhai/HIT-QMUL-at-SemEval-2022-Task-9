#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/12/15 10:14
# @Author  : hit-itnlp-fengmq
# @FileName: sub2_eva.py
# @Software: PyCharm

# 3080:  /home/mqfeng/code/RecipeQA/sub2

import json
import os
import pickle
import re
from collections import namedtuple

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM

os.environ["CUDA_VISIBLE_DEVICES"] = '1'


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
        newans = []
        for tag in tags:
            if 'habitat' in tag or 'tool' in tag:
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
        if 'habitat' in tags or 'tool' in tags:
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


def get_premodel(path=r"D:\Anaconda\learn\_Bert\pre_train_model\albert-base-v2"):
    '''
    :param path: 预训练模型在本机的路径
    :return:
    '''
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForMaskedLM.from_pretrained(path)  # Embedding(30000, 128) [MASK]:4
    print("model load end!")
    return model, tokenizer


def get_text_from_docid(docid: str):
    with open("test_all_text_tag.json") as f:
        texts = json.load(f)
    text = []
    for item in texts[docid]['text_tags']:
        text.append(item["text"])
    return text


def get_textAndTag_from_docid(docid: str, Stexts: list):
    with open("test_all_text_tag.json") as f:
        texts = json.load(f)
    temp = []
    for item in texts[docid]['text_tags']:
        if item["text"] in Stexts:
            temp_d = {}
            temp_d["text"] = item["text"]
            temp_d["hidden"] = item["tagcols"]["hidden"]
            temp.append(temp_d)
    return temp


def scores(question: str, text: list, model, top_k: int = 4):
    text_embeddings = model.encode(text, convert_to_tensor=True)
    query_embedding = model.encode(question, convert_to_tensor=True)

    if torch.cuda.is_available():
        text_embeddings.to('cuda')
        query_embedding = torch.unsqueeze(query_embedding, dim=0)
        query_embedding.to('cuda')
        text_embeddings = util.normalize_embeddings(text_embeddings)
        query_embedding = util.normalize_embeddings(query_embedding)
    top_results = util.semantic_search(query_embedding, text_embeddings
                                       , top_k=top_k
                                       )  # [[{corpus_id:,score:},{},{}]]
    text_ids = [item['corpus_id'] for item in top_results[0]]
    # 还原顺序
    text_ids.sort()
    result = []
    for id in text_ids:
        result.append(text[id])
    return result


def pre_process(data):
    # [{"text":"","hidden":[]},{},{}]
    new_data = []
    for item in data:
        texts = item['text']
        hidden = item['hidden']
        new_tags = []
        for tg in hidden:
            if tg == "_":
                new_tags.append("_")
            else:
                tg = "|".join(tg).replace(":", "=")
                t_tg = useful_tag_8(tg)
                if t_tg != "_":
                    temp = "# " + (" # ".join(t_tg)) + " #"
                else:
                    temp = (" # ".join(t_tg))
                new_tags.append(temp)
        if new_tags.count("_") == len(new_tags):  # 说明tag全部为_，直接过滤掉
            continue
        temp_dict = {}
        temp_dict['text'] = texts
        temp_dict['tags'] = new_tags
        new_data.append(temp_dict)

    text, tag = [], []
    for item_text in new_data:
        text.extend(item_text['text'].strip(" ").split(" "))
        tag.extend(item_text['tags'])
    if len(text) != len(tag):
        print("length not equal!")
    new_text = []
    for i in range(len(text)):
        if tag[i] == "_":
            new_text.append(text[i])
        else:
            new_text.append(text[i] + " " + tag[i])
    text = " ".join(new_text)
    return text


class MyModel(nn.Module):
    def __init__(self, albert, QAhead):
        super().__init__()
        self.albert = albert

        self.qa_outputs = QAhead  # self.qa_outputs = nn.Linear(768, 2,bias=True)

    def forward(self, input_ids, token_type_ids, attention_mask, start_positions=None, end_positions=None):
        albert_output = self.albert(input_ids=input_ids,
                                    token_type_ids=token_type_ids,
                                    attention_mask=attention_mask)
        last_hidden_state = albert_output.last_hidden_state  # torch.Size([batchSize, 512, 768]) 512与max_length相关
        logits = self.qa_outputs(last_hidden_state)  # torch.Size([batchSize, 512, 2])
        start_logits, end_logits = logits.split(1, dim=-1)  # 分离出的start_logits/end_logits形状为([batchSize, 512, 1])
        start_logits = start_logits.squeeze(-1)  # ([batchSize, 512])
        end_logits = end_logits.squeeze(-1)  # ([batchSize, 512])

        Outputs = namedtuple('Outputs', 'start_logits, end_logits')
        outputs = Outputs(start_logits, end_logits)
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)
            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            Outputs = namedtuple('Outputs', 'loss,start_logits, end_logits')
            outputs = Outputs(total_loss, start_logits, end_logits)
        return outputs


model, tokenizer = get_premodel(path=r'/home/mqfeng/code/RecipeQA/pretrain_model/epoch2')
with open(r'/home/mqfeng/code/RecipeQA/pretrain_model/QAhead_large.pickle', 'rb') as file:
    QAhead = pickle.load(file)
# myModel = MyModel(model.albert,QAhead)
# myModel.load_state_dict(torch.load(r'/home/mqfeng/code/RecipeQA/sub2/save3/model_3.pth'))
myModel = torch.load(r'/home/mqfeng/code/RecipeQA/sub2/save3/model_3.pth')
Sbert = SentenceTransformer(r'/home/mqfeng/pretrainModel/all-MiniLM-L6-v2')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
myModel.to(device)

with open("start2.json") as f:
    data2 = json.load(f)
result = {}
with torch.no_grad():
    for item in tqdm(data2):
        q_id = item['q_id'].strip()
        question = item['question'].strip()
        doc_id = item['doc_id'].strip()

        texts = get_text_from_docid(doc_id)
        texts = scores(question, texts, Sbert)
        texts = get_textAndTag_from_docid(doc_id, texts)  # [{"text":"","hidden":[]},{},{}]
        texts = pre_process(texts)

        encode = tokenizer.encode_plus(question, texts, add_special_tokens=True,
                                       max_length=512, padding='max_length',
                                       return_attention_mask=True, return_tensors='pt',
                                       truncation=True)
        input_ids, token_type_ids, attention_mask = encode['input_ids'], encode['token_type_ids'], encode[
            'attention_mask']
        outputs = myModel(input_ids.to(device), token_type_ids=token_type_ids.to(device),
                          attention_mask=attention_mask.to(device))
        start_pred = torch.argmax(outputs.start_logits, dim=1)
        end_pred = torch.argmax(outputs.end_logits, dim=1)
        ans = input_ids[0][start_pred.item():end_pred.item() + 1]
        ans = tokenizer.decode(ans)

        if question.startswith("how"):
            ans = "by using a " + ans
        if doc_id not in result.keys():
            result[doc_id] = {}
        result[doc_id][q_id] = ans
fw = open("sub2_ans_error.json", "w", encoding="utf-8")
json.dump(result, fw, ensure_ascii=False, indent=4)
fw.close()
