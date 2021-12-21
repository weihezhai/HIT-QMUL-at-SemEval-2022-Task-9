#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/12/12 19:43
# @Author  : hit-itnlp-fengmq
# @FileName: eva.py
# @Software: PyCharm
import json

if __name__ == "__main__":
    result = {}
    with open(r"test/test_q.json", "r", encoding="utf-8") as f:
        json_datas = json.load(f)
    for qa_item in json_datas:
        doc_id = qa_item['doc_id']
        q_id = qa_item['q_id']
        question = qa_item['question']
        ans = None
        if q_id.startswith("0"):
            # 0 number 类
            pass
        elif q_id.startswith("2"):
            # 第二类精准匹配
            pass
        elif q_id.startswith("4"):
            # 4 first/second
            pass
        elif q_id.startswith("18"):
            # 18 N/A问题
            ans = "N/A"
        elif q_id.startswith("1-") or q_id.startswith("3") or q_id.startswith("6") or q_id.startswith(
                "10") or q_id.startswith("14") or q_id.startswith("17"):
            # 生成式
            pass
        else:
            # 5，7，8，9，11，12，13，15，16精准匹配
            pass

        if doc_id not in result.keys():
            result[doc_id] = {}
        result[doc_id][q_id] = ans
    with open("r2vq_pred.json", 'w', encoding='utf-8') as w:
        json.dump(result, w, ensure_ascii=False, indent=4)
        w.close()
