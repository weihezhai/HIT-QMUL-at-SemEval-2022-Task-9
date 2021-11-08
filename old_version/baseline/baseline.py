#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/10/19 22:24
# @Author  : fengmq
# @FileName: bert4QA_baseline.py
# @Software: PyCharm

# 按照知乎代码微调  参考：https://zhuanlan.zhihu.com/p/357528657
# /home/mqfeng/.conda/envs/fmq/bin/python bert4QA_baseline.py
import os

import torch
import numpy as np
import random
from tqdm import tqdm
import json
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW
from transformers import WEIGHTS_NAME, CONFIG_NAME
from transformers import get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
# 随机数种子
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
setup_seed(2021)

# 加载模型
def get_premodel(path=r"E:\RecipeQA\transformers_models\bert-large-uncased-whole-word-masking-finetuned-squad"):
    '''
    :param path: 预训练模型在本机的路径  下载链接（https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad/tree/main）
    :return:
    '''
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForQuestionAnswering.from_pretrained(path)
    print("model load end!")
    return model,tokenizer

# Dataset
def split_data(path=r'text_data.json',fold=0):
    '''
    总共1000个食谱，10202条问答对
    :param path: text_data.json的路径
    :param fold: 第几折，默认总共五折交叉
    :return: 返回划分好的训练集与验证集,是一个list，list的元素为一个字典，包含{qa：''，text：''}，其中qa以四个空格分割，q,a=qa.split("     ")
    '''
    datas=[]
    f = open(path, 'r', encoding='utf-8')
    json_data = json.load(f)
    f.close()

    for key in json_data:
        item = json_data[key]
        question = item['question']
        text = item['text']
        for qa in question:
            q,a=qa.split("     ")
            if a in text:
                data={}
                data['qa']=qa
                data['text'] = text
                datas.append(data)
    num_data = len(datas)

    num_test = num_data//5
    random.shuffle(datas)
    test = datas[fold * num_test:(fold + 1) * num_test]
    if fold == 0:
        train = datas[num_test:]
    else:
        train = datas[:num_test * fold]
        train.extend(datas[num_test * (fold + 1):])
    print("nums of train data:{}".format(len(train)))
    print("nums of val data:{}".format(len(test)))
    return train, test


class myDataset(Dataset):  # 需要继承data.Dataset
    def __init__(self, data,tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        item = self.data[index]
        qa,text = item['qa'],item['text']
        q, a = qa.split("     ")
        # encode= self.tokenizer(q, text, add_special_tokens=True, return_tensors="pt")
        encode = self.tokenizer.encode_plus(q, text, add_special_tokens=True,
                                            max_length=512, padding='max_length',
                                            return_attention_mask=True, return_tensors='pt',
                                            truncation=True)
        input_ids, token_type_ids, attention_mask = encode['input_ids'],encode['token_type_ids'],encode['attention_mask']
        # 获取起始位置
        start,end = self.start_end(a,q,text)
        return  input_ids.squeeze(), token_type_ids.squeeze(), attention_mask.squeeze(),torch.tensor(start),torch.tensor(end)

    def __len__(self):
        return len(self.data)

    def start_end(self,answer,q,text):
        # 有问题
        answer_encode = self.tokenizer(answer)['input_ids'][1:-1]
        text_encode = self.tokenizer(q, text)['input_ids']
        start_end=()
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
        if len(start_end)==0:
            start_end=(0,0)
        return start_end

def train(model,tokenizer, train_dataloader, testdataloader,device,fold,epoch=3):
    model.to(device)
    optim = AdamW(model.parameters(), lr=1e-5, weight_decay=0.1)
    scheduler = get_linear_schedule_with_warmup(
        optim, num_warmup_steps=400, num_training_steps=len(train_dataloader) * epoch)

    for epoch in range(epoch):
        model.train()
        train_acc=[]
        train_loss=[]
        loop = tqdm(train_dataloader, leave=True)
        for data in tqdm(loop):
            optim.zero_grad()
            data = tuple(t.to(device) for t in data)
            input_ids, token_type_ids, attention_mask,start,end= data

            outputs = model(input_ids,token_type_ids=token_type_ids, attention_mask=attention_mask,start_positions=start,end_positions=end)
            loss, start_logits, end_logits = outputs["loss"],outputs["start_logits"],outputs['end_logits']
            loss.backward()
            optim.step()
            scheduler.step()

            start_pred = torch.argmax(start_logits, dim=1)
            end_pred = torch.argmax(end_logits, dim=1)

            acc = ((start_pred == start).sum() / len(start_pred)).item()
            train_acc.extend((start_pred == start).int().tolist())
            train_acc.extend((end_pred == end).int().tolist())
            train_loss.append(loss.item())

            loop.set_description(f'fold:{fold}  Epoch:{epoch}')
            loop.set_postfix(loss=loss.item(),acc=acc)
        #保存模型
        # model_to_save = model.module if hasattr(model, 'module') else model
        # model_path = r"/home/mqfeng/R2QA/models/fold" + str(fold) + "_epoch" + str(epoch)
        # if not os.path.exists(model_path):
        #     os.mkdir(model_path)
        # output_model_file = os.path.join(model_path, WEIGHTS_NAME)
        # output_config_file = os.path.join(model_path, CONFIG_NAME)
        # torch.save(model_to_save.state_dict(), output_model_file)
        # model_to_save.config.to_json_file(output_config_file)
        # tokenizer.save_vocabulary(model_path)
        # print("fold: {},epoch {} saved!".format(fold,epoch))

        model.eval()
        test_acc = []
        test_loss=[]
        for data in tqdm(testdataloader):
            with torch.no_grad():
                data = tuple(t.to(device) for t in data)
                input_ids, token_type_ids, attention_mask, start, end = data
                outputs = model(input_ids,token_type_ids=token_type_ids, attention_mask=attention_mask,start_positions=start,end_positions=end)
                loss = outputs["loss"]
                start_pred = torch.argmax(outputs['start_logits'], dim=1)
                end_pred = torch.argmax(outputs['end_logits'], dim=1)

                test_acc.extend(((start_pred == start).int().tolist()))
                test_acc.extend(((end_pred == end).int().tolist()))
                test_loss.append(loss.item())
        print("{},Train_acc:{} Train_loss:{}-----Val_acc:{} Val_loss:{}".format(epoch,np.mean(train_acc),np.mean(train_loss),np.mean(test_acc),np.mean(test_loss)))

for fold in range(5):
    # 修改预训练模型所在的路径：下载链接（https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad/tree/main）
    # 新下载的pytorch_model.bin可能文件名不正确，需要重命名为pytorch_model.bin
    model,tokenizer = get_premodel(r"bert-large-uncased-whole-word-masking-finetuned-squad")#
    # 划分五折交叉数据  #修改为本机路径
    train_data, test_data = split_data(r"text_data.json",fold)
    # 构造DataSet和DataLoader
    train_Dataset = myDataset(train_data,tokenizer)
    test_Dataset = myDataset(test_data,tokenizer)
    # 修改batchsize
    train_Dataloader = DataLoader(train_Dataset, batch_size=2,shuffle=True)
    test_Dataloader = DataLoader(test_Dataset, batch_size=1)
    # 训练
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    print("----------------fold {} begin ----------------------".format(fold))
    train(model,tokenizer,train_Dataloader,test_Dataloader,device,fold,epoch=3)