utils.py：工具库
text_data3.json: 10_28号版本的问答对数据。将食谱当中相似度最高的几句话与问题关联起来，提取关联度最高的三句话

AlMLMQA.py:利用Albert模型在recipeQA数据集的食谱数据上进行了MLM，领域内训练后。
            在R2QA上进行训练，（训练答案只会出现在句子当中的），
            这是没有进行文本与问题相似度比较的基础上进行运行的代码。

AlSimQA.py: 利用text_data2.json训练的答案
QAhead_base.pickle和QAhead_large.pickle是Albert_base_v2和Albert_large_v2的答案问答头部。
log文件在base和large上的训练日志
