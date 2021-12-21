import json
import os
import pickle
import random
import pprint
import time
import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split, RandomSampler, SequentialSampler
from tqdm import tqdm
from transformers import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForQuestionAnswering
from transformers import get_linear_schedule_with_warmup
import torch.nn.functional as F
from collections import namedtuple
from number_classification import get_premodel

l='allenai/longformer-large-4096-finetuned-triviaqa'
b='google/bigbird-roberta-large'
p='google/bigbird-pegasus-large-arxiv'

model, tokenizer = get_premodel(p)
model.load_state_dict(torch.load(''))
