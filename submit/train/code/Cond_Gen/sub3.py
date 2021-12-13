#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/12/7 13:02
# @Author  : hit-itnlp-fengmq
# @FileName: sub3.py.py
# @Software: PyCharm
# 第1,3，6 10 14 ,17六类问题在seed=21下78
# 保存在/data/home/acw664/Cond_Gen/save
import json
import os
import random
import numpy as np
import torch
import torch.nn as nn
# from sentence_transformers import util, SentenceTransformer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration, Adafactor, get_linear_schedule_with_warmup

#os.environ["CUDA_VISIBLE_DEVICES"] = '1'


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


setup_seed(21)


# 加载模型
def get_premodel(path=r"D:\Anaconda\learn\_Bert\pre_train_model\t5-small"):
    '''
    :param path: 预训练模型在本机的路径
    :return:
    '''
    tokenizer = T5Tokenizer.from_pretrained(path)
    model = T5ForConditionalGeneration.from_pretrained(path)
    print("model load end!")
    return model, tokenizer


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


def split_data(path=r'3_all_data.json', is_topk=False, Sbert=None, fold=0):
    '''
    {'ques': " What's in the egg white?",
     'answer': ' the eggs',
     'sim_texts': [{'text': ' Preheat oven to 375 degF ( 190 degC ) .',
                   'hidden': ['_', '_', '_', '_', '_', '_', '_', '_', '_', '_'],
                   'patient': []}]
    }
    '''
    f = open(path, 'r', encoding='utf-8')
    json_data = json.load(f)
    f.close()
    print("split data begin !")
    datas = []
    for items in tqdm(json_data):
        list_qa = items['list_2qa']
        text_tags = items['text_tags']
        all_recipe_texts = [i['text'] for i in text_tags]  # 从question_tag得到所有的文本信息。
        for qa in list_qa:
            ques, answer = qa['quest'], qa['answer']

            if is_topk and Sbert:  # 利用sentence Bert提取前k句话
                sim_texts = scores(ques, all_recipe_texts, Sbert)  # 使用sentence Bert得到最相似的k句话
                sim_texts_tags = []  # 存储sim_text和它对应的tags
                for st in sim_texts:  # 将每一句得到tag
                    for tts in text_tags:
                        if tts['text'] == st:
                            sim_texts_tags.append(tts)
                            break
                sub_item_dict = {}
                sub_item_dict["ques"] = ques
                sub_item_dict["answer"] = answer
                sub_item_dict["sim_texts"] = sim_texts_tags
                datas.append(sub_item_dict)
            else:
                sub_item_dict = {}
                sub_item_dict["ques"] = ques
                sub_item_dict["answer"] = answer
                sub_item_dict["sim_texts"] = text_tags
                datas.append(sub_item_dict)
    num_data = len(datas)
    num_test = num_data // 10
    random.shuffle(datas)

    test = datas[fold * num_test:(fold + 1) * num_test]
    if fold == 0:
        train = datas[num_test:]
    else:
        train = datas[:num_test * fold]
        train.extend(datas[num_test * (fold + 1):])
    print("split data end !")

    print("nums of train data:{}".format(len(train)))
    print("nums of val data:{}".format(len(test)))
    return train, test


def pre_process(data):
    '''
    将tag进行处理，例如Tool=spatula.2.3.1，直接处理为spatula，
    但是若同时存在Habitat ,Tool，会取优先出现的，
    同时过滤tag全部为_的句子，
    '''

    def useful_tag(tags: str):
        tags = tags.lower()
        if "|" in tags:
            tags = tags.split("|")
            ans = []
            for tag in tags:
                if 'habitat' in tag or 'tool' in tag or 'result' in tag or 'drop' in tag:
                    name, tag = tag.split("=")
                    if ":" not in tag:
                        tag = tag.split(".")[0]
                        if '_' in tag:  # medium_oven_-_proof_skillet
                            tag = tag.split("_")
                            tag = " ".join(tag)
                    else:#Drop=flour_mixture.4.1.17:eggs.4.1.2:dough.4.1.15
                        temp_tag=[]
                        tag = tag.split(":")
                        for t_t in tag:
                            thing = t_t.split(".")[0]
                            if "_" in thing:
                                thing = thing.split("_")
                                thing = " ".join(thing)
                            temp_tag.append(thing)
                        tag=" , ".join(temp_tag)
                    ans.append(name + " : " + tag)
            if len(ans) != 0:
                return ans
            return "_"
        else:
            ans = []
            if 'habitat' in tags or 'tool' in tags or  'result' in tags or 'drop' in tags:
                name, tag = tags.split("=")
                if ":" not in tag:
                    tag = tag.split(".")[0]
                    if '_' in tag:  # medium_oven_-_proof_skillet
                        tag = tag.split("_")
                        tag = " ".join(tag)
                    ans.append(name + " : " + tag)
                    return ans
                else:#Drop=flour_mixture.4.1.17:eggs.4.1.2:dough.4.1.15
                    tag = tag.split(":")
                    temp_tag=[]
                    for t_t in tag:
                        thing = t_t.split(".")[0]
                        if "_" in thing:
                            thing = thing.split("_")
                            thing = " ".join(thing)
                        temp_tag.append(thing)
                    tag=" , ".join(temp_tag)
                    ans.append(name + " : " + tag)
                    return ans
                    pass
            else:
                return "_"

    print("pre_process data start !")
    # Habitat ,Tool
    new_data = []
    for item in data:
        sim_texts = item['sim_texts']
        new_sim_texts = []  # 更换sim_texts
        for i in sim_texts:
            text = i['text'].lower()
            hiddens = i['hidden']  # 第三类问题有一个hidden，patient
            patients = i['patient']

            new_hiddens = []
            new_patients = []

            for tg in hiddens:
                if tg == "_":
                    new_hiddens.append("_")
                else:
                    t_tg = useful_tag(tg)
                    if t_tg != "_":
                        temp = "# " + (" # ".join(t_tg)) + " #"
                    else:
                        temp = (" # ".join(t_tg))
                    new_hiddens.append(temp)

            for tg in patients:
                if tg == "_":
                    new_patients.append("_")
                else:
                    tg = tg.lower()
                    # if tg in ["i-patient","b-patient","b-instrument","i-instrument","b-attribute","i-attribute","b-goal","i-goal"]:
                    if tg in ["i-patient", "b-patient"]:
                        temp = "# " + tg + " #"
                    else:
                        temp = '_'
                    new_patients.append(temp)
            # 说明tag全部为_，直接过滤掉
            # if new_hiddens.count("_") == len(new_tags):
            #    continue
            temp_dict = {}
            temp_dict['text'] = text
            temp_dict['hiddens'] = new_hiddens
            temp_dict['patients'] = new_patients
            new_sim_texts.append(temp_dict)

        item["sim_texts"] = new_sim_texts
        new_data.append(item)
    print("pre_process data end !")
    return new_data


class myDataset(Dataset):  # 需要继承data.Dataset
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

        self.max_source_length = 512
        self.max_target_length = 64

    def __getitem__(self, index):
        item = self.data[index]
        q = item['ques'].strip(" ").lower()
        a = item['answer'].strip(" ")
        texts = item['sim_texts']

        # 处理text和tag
        text, hiddens, patients = [], [], []
        for item_text in texts:
            text += item_text['text'].strip(" ").split(" ")
            hiddens += item_text['hiddens']
            patients += item_text['patients']
        if len(text) != len(hiddens):
            print("hiddens length not equal!")
        if len(text) != len(patients):
            print("patients length not equal!")

        new_text = []
        for i in range(len(text)):
            text_i = text[i]
            if hiddens[i] != "_":
                text_i += (" " + hiddens[i])
            if patients[i] != "_":
                text_i += (" " + patients[i])
            new_text.append(text_i)
        text = " ".join(new_text)
        # 问题和文本
        q_text = q + " : " + text
        source_encoding = self.tokenizer.encode_plus(q_text,
                                                     add_special_tokens=True,
                                                     max_length=self.max_source_length,
                                                     padding='max_length',
                                                     return_attention_mask=True,
                                                     return_tensors='pt',
                                                     truncation=True)
        input_ids, attention_mask = source_encoding['input_ids'], source_encoding['attention_mask']
        # 答案
        target_encoding = self.tokenizer.encode_plus(a,
                                                     add_special_tokens=True,
                                                     max_length=self.max_target_length,
                                                     padding='max_length',
                                                     return_attention_mask=True,
                                                     truncation=True)
        labels = target_encoding.input_ids
        labels = [(label if label != tokenizer.pad_token_id else -100) for label in labels]
        labels = torch.tensor(labels)

        return input_ids.squeeze(), attention_mask.squeeze(), labels

    def __len__(self):
        return len(self.data)


def train(model, train_Dataloader, test_Dataloader, device, fold=0, epochs=5):
    model.to(device)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    optimizer = Adafactor(model.parameters(),
                          lr=2e-4,
                          eps=(1e-30, 1e-3),
                          clip_threshold=1.0,
                          decay_rate=-0.8,
                          beta1=None,
                          weight_decay=0.0,
                          relative_step=False,
                          scale_parameter=False,
                          warmup_init=False)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=200, num_training_steps=len(train_Dataloader) * epochs)
    for epoch in range(epochs):
        model.train()
        loop = tqdm(train_Dataloader, leave=True)
        train_loss = []
        for data in tqdm(loop):
            optimizer.zero_grad()
            data = tuple(t.to(device) for t in data)
            input_ids, attention_mask, labels = data
            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = output.loss.mean()
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss.append(loss.item())
            loop.set_description(f'fold:{fold}  Epoch:{epoch}')
            loop.set_postfix(loss=loss.item())
        if epoch>=4:
            s_path = r"/data/home/acw664/Cond_Gen/save"
            sub_path = os.path.join(s_path, "model" + str(epoch))
            os.mkdir(sub_path)
            model.module.save_pretrained(sub_path)

        model.eval()
        test_acc = []
        test_loss = []
        test_ans=[]
        with torch.no_grad():
            for data in tqdm(test_Dataloader):
                data = tuple(t.to(device) for t in data)
                input_ids, attention_mask, labels = data
                if torch.cuda.device_count() > 1:
                    outputs = model.module.generate(input_ids)
                else:
                    outputs = model.generate(input_ids)
                output_all = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

                test_loss.append(output_all.loss.sum().cpu())

                labels = [[i.item() for i in label if i != -100] for label in labels]
                decode_labels = tokenizer.decode(labels[0], skip_special_tokens=True)
                decode_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
                temp_dict={}
                if decode_labels == decode_output:
                    test_acc.append(1)
                    if epoch==4:
                        temp_dict["label"]=decode_labels
                        temp_dict["output"] = decode_output
                        temp_dict['is']=1
                else:
                    test_acc.append(0)
                    if epoch==4:
                        temp_dict["label"]=decode_labels
                        temp_dict["output"] = decode_output
                        temp_dict['is']=0
                if epoch == 4:
                    test_ans.append(temp_dict)
        print("{}，Train_loss:{}----Test_loss:{} Test_acc:{}".format(epoch, np.mean(train_loss), np.mean(test_loss),
                                                                    np.mean(test_acc)))
        if epoch == 4:
            fw = open("ans.json", "w", encoding="utf-8")
            json.dump(test_ans, fw, ensure_ascii=False, indent=4)
            fw.close()


for fold in range(1):
    model, tokenizer = get_premodel("t5-large")

    # Sbert = SentenceTransformer(r'/home/mqfeng/preModels/all-MiniLM-L6-v2')
    train_data, test_data = split_data("six_all_data.json")
    train_data, test_data = pre_process(train_data), pre_process(test_data)


    train_Dataset = myDataset(train_data, tokenizer)
    test_Dataset = myDataset(test_data, tokenizer)
    train_Dataloader = DataLoader(train_Dataset, batch_size=4, shuffle=True)
    test_Dataloader = DataLoader(test_Dataset, batch_size=1)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train(model, train_Dataloader, test_Dataloader, device, fold, epochs=7)
#73%的准确率
