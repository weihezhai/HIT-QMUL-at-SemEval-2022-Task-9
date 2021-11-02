#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/10/17 21:46
# @Author  : fengmq
# @FileName: tsm_qa.py
# @Software: PyCharm
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

def get_premodel(path=r"E:\RecipeQA\transformers_models\bert-large-uncased-whole-word-masking-finetuned-squad"):
    '''
    :param path: 预训练模型在本机的路径  下载链接（https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad/tree/main）
    :return:
    '''
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForQuestionAnswering.from_pretrained(path)
    print("model load end!")
    return model,tokenizer

def QA(model,tokenizer,questions,texts):
    """
    :param model: 预训练bert模型
    :param tokenizer:  分词器
    :param question:  问题， 注意：是列表的形式，可以有多个问题
    :param texts: 文本  str形式
    :return: 答案
    """
    print("=" * 20)
    print(f"text:{texts}\n")
    for question in questions:
        inputs = tokenizer(question, texts, add_special_tokens=True, return_tensors="pt")
        input_ids = inputs["input_ids"].tolist()[0]

        outputs = model(**inputs)
        answer_start_scores = outputs.start_logits
        answer_end_scores = outputs.end_logits
        # Get the most likely beginning of answer with the argmax of the score
        answer_start = torch.argmax(answer_start_scores)
        # Get the most likely end of answer with the argmax of the score
        answer_end = torch.argmax(answer_end_scores) + 1
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
        print(f"Question: {question}")
        print(f"Answer: {answer}")
        print("-" * 20)
    pass
model,tokenizer = get_premodel()
text="Follow the instructions on the back of your cookie mix. " \
      "For this specific package,we started by pouring the mix into a large mixing bowl." \
      " Next, we added 1 stick of softened butter to the mix. Lastly, we added 1 egg." \
      "We then mixed the dough ingredients together until it was a soft, doughy consistency."
question=["what do we add eggs to ?","how many eggs should we add ?","what should eggs be added to ?"]
QA(model,tokenizer,question,text)
