recipe_corpora.txt 是RecipeQA数据集的文本，用于增量预训练模型。
text_data2.json： 从数据集提取出问答对的json文件
将其划分为三个文件：event.json:number、json、other.json。需要注意的是答案为N/A的混杂在三种类型当中
bert4QA_baseline.py是利用hugging face的bert-large-uncased-whole-word-masking-finetuned-squad进行的baseline模型