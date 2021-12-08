#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/11/25 15:12
# @Author  : hit-itnlp-fengmq
# @FileName: sub2_question_merge.py
# @Software: PyCharm
# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/11/25 12:40
# @Author  : hit-itnlp-fengmq
# @FileName: sub2_question_concat.py
# @Software: PyCharm
import json
import pickle
import random
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer, util
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AdamW
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import get_linear_schedule_with_warmup

# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


setup_seed(2021)


# 加载模型
def get_premodel(path=r"D:\Anaconda\learn\_Bert\pre_train_model\albert-base-v2"):
    '''
    :param path: 预训练模型在本机的路径
    :return:
    '''
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForMaskedLM.from_pretrained(path)  # Embedding(30000, 128) [MASK]:4
    print("model load end!")
    return model, tokenizer


def scores(question: str, text: list, model, top_k: int = 3):
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


def split_data(path=r'2_all_data.json', is_topk=False, Sbert=None, fold=0):
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
    num_test = num_data // 5
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
                if 'habitat' in tag or 'tool' in tag:
                    name, tag = tag.split("=")
                    tag = tag.split(".")[0]

                    if '_' in tag:  # medium_oven_-_proof_skillet
                        tag = tag.split("_")
                        tag = " ".join(tag)
                    ans.append(name + " : " + tag)
            if len(ans) != 0:
                return ans
            return "_"
        else:
            ans = []
            if 'habitat' in tags or 'tool' in tags:
                name, tag = tags.split("=")
                tag = tag.split(".")[0]
                if '_' in tag:  # medium_oven_-_proof_skillet
                    tag = tag.split("_")
                    tag = " ".join(tag)
                ans.append(name + " : " + tag)
                return ans
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
            tags = i['tags']  # 第二类问题只有一个tags
            new_tags = []
            for tg in tags:
                if tg == "_":
                    new_tags.append("_")
                else:
                    t_tg = useful_tag(tg)
                    if t_tg != "_":
                        temp = "# " + (" # ".join(t_tg)) + " #"
                    else:
                        temp = (" # ".join(t_tg))
                    new_tags.append(temp)
            if new_tags.count("_") == len(new_tags):  # 说明tag全部为_，直接过滤掉
                continue
            temp_dict = {}
            temp_dict['text'] = text
            temp_dict['tags'] = new_tags
            new_sim_texts.append(temp_dict)

        item["sim_texts"] = new_sim_texts
        new_data.append(item)
    print("pre_process data end !")
    return new_data


class myDataset(Dataset):  # 需要继承data.Dataset
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        item = self.data[index]
        q = item['ques'].strip(" ").lower()
        a = item['answer'].strip(" ")
        texts = item['sim_texts']
        # 处理答案，分离两种
        if "by using a" in a:
            a = a.split(" ")[3:]
            a = " ".join(a)
        # 处理text和tag
        text, tag = [], []
        for item_text in texts:
            text += item_text['text'].strip(" ").split(" ")
            tag += item_text['tags']
        if len(text) != len(tag):
            print("length not equal!")
        new_text = []
        for i in range(len(text)):
            if tag[i] == "_":
                new_text.append(text[i])
            else:
                new_text.append(text[i] + " " + tag[i])
        text = " ".join(new_text)
        ##q,a,text,tag
        # encode= self.tokenizer(q, text+tag, add_special_tokens=True, return_tensors="pt")
        encode = self.tokenizer.encode_plus(q,
                                            text,
                                            add_special_tokens=True,
                                            max_length=512,
                                            padding='max_length',
                                            return_attention_mask=True,
                                            return_tensors='pt',
                                            truncation=True)
        input_ids, token_type_ids, attention_mask = encode['input_ids'], encode['token_type_ids'], encode[
            'attention_mask']
        # 获取起始位置
        start, end = self.start_end(a, q, text)
        return input_ids.squeeze(), token_type_ids.squeeze(), attention_mask.squeeze(), torch.tensor(
            start), torch.tensor(end)

    def __len__(self):
        return len(self.data)

    def start_end(self, answer, q, text):
        # 有问题
        answer_encode = self.tokenizer(answer.strip().lower())['input_ids'][1:-1]
        text_encode = self.tokenizer(q.strip().lower(), text.strip().lower())['input_ids']
        start_end = ()
        for i in range(len(text_encode)):
            if text_encode[i] == answer_encode[0]:
                j = 0
                for j in range(len(answer_encode)):
                    if text_encode[i + j] != answer_encode[j]:
                        break
                if j == len(answer_encode) - 1:
                    if text_encode[i + j] != answer_encode[j]:
                        continue
                    start_end = (i, i + len(answer_encode) - 1)
            if len(start_end) != 0:
                return start_end
        if len(start_end) == 0:
            start_end = (0, 0)
        return start_end


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


def train(model, train_dataloader, testdataloader, device, fold=0, epoch=5):
    model.to(device)
    optim = AdamW(model.parameters(), lr=1e-5, weight_decay=0.2)
    scheduler = get_linear_schedule_with_warmup(
        optim, num_warmup_steps=400, num_training_steps=len(train_dataloader) * epoch)

    for epoch in range(epoch):
        model.train()
        train_acc2 = []
        train_loss = []
        loop = tqdm(train_dataloader, leave=True)
        for data in tqdm(loop):
            optim.zero_grad()
            data = tuple(t.to(device) for t in data)
            input_ids, token_type_ids, attention_mask, start, end = data

            outputs = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                            start_positions=start, end_positions=end)
            loss, start_logits, end_logits = outputs.loss, outputs.start_logits, outputs.end_logits

            loss.backward()
            optim.step()
            scheduler.step()

            start_pred = torch.argmax(start_logits, dim=1)
            end_pred = torch.argmax(end_logits, dim=1)

            acc = ((start_pred == start).sum() / len(start_pred)).item()
            starts = (np.array((start_pred == start).cpu()))
            ends = (np.array((end_pred == end).cpu()))
            acc2 = np.all(np.array([starts, ends]).T, axis=-1).astype(int)
            train_acc2.extend(acc2)
            train_loss.append(loss.item())

            loop.set_description(f'fold:{fold}  Epoch:{epoch}')
            loop.set_postfix(loss=loss.item(), acc=acc)
        # if epoch >=3:
        #     model_to_save = model.module if hasattr(model, 'module') else model
        #     model_path = r"fold" + str(fold) + "_epoch" + str(epoch)
        #     if not os.path.exists(model_path):
        #         os.makedirs(model_path)
        #     torch.save(model_to_save, os.path.join(model_path,'model.pt'))

        model.eval()
        test_loss = []
        test_acc2 = []
        for data in tqdm(testdataloader):
            with torch.no_grad():
                data = tuple(t.to(device) for t in data)
                input_ids, token_type_ids, attention_mask, start, end = data
                outputs = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                                start_positions=start, end_positions=end)

                loss = outputs.loss
                start_pred = torch.argmax(outputs.start_logits, dim=1)
                end_pred = torch.argmax(outputs.end_logits, dim=1)

                starts = (np.array((start_pred == start).cpu()))
                ends = (np.array((end_pred == end).cpu()))
                acc2 = np.all(np.array([starts, ends]).T, axis=-1).astype(int)
                test_acc2.extend(acc2)
                test_loss.append(loss.item())
        print("{}, Train_acc2:{} Train_loss:{}-----Val_acc:{}  Val_loss:{}".format(
            epoch, np.mean(train_acc2), np.mean(train_loss), np.mean(test_acc2),
            np.mean(test_loss)))


for fold in range(1):
    #修改模型路径，以及对应的QAhead.pickle。根据设备酌情修改batchsize
    model, tokenizer = get_premodel(r'/home/mqfeng/R2QA/pretrain_recipeQA/large/epoch2')
    with open(r'QAhead_large.pickle', 'rb') as file:
        QAhead = pickle.load(file)
    myModel = MyModel(model.albert, QAhead)
    Sbert = SentenceTransformer(r'/home/mqfeng/preModels/all-MiniLM-L6-v2')
    train_data, test_data = split_data(is_topk=True, Sbert=Sbert)
    train_data, test_data = pre_process(train_data), pre_process(test_data)

    # 构造DataSet和DataLoader
    train_Dataset = myDataset(train_data, tokenizer)
    test_Dataset = myDataset(test_data, tokenizer)
    # 修改batchsize
    train_Dataloader = DataLoader(train_Dataset, batch_size=4, shuffle=True)
    test_Dataloader = DataLoader(test_Dataset, batch_size=1)
    # 训练
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    train(myModel, train_Dataloader, test_Dataloader, device, fold, epoch=5)
