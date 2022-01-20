# This training code is based on the `run_glue.py` script from huggingface.
import json
import os
import pickle
import random
import pprint
import time
import datetime
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split, RandomSampler, SequentialSampler
from tqdm import tqdm
from transformers import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, T5Tokenizer, T5ForConditionalGeneration
from transformers import get_linear_schedule_with_warmup
import torch.nn.functional as F
from collections import namedtuple

# from sentence_transformers import SentenceTransformer, util
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


setup_seed(2021)


def flat_accuracy(preds, labels):
    '''
    Function to calculate the accuracy of our predictions vs labels
    '''
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def train_summary(data):  # save summary to csv file.
    pd.set_option('precision', 2)

    # Create a DataFrame from our training statistics.
    df_stats = pd.DataFrame(data)
    # Use the 'epoch' as the row index.
    df_stats = df_stats.set_index('epoch')
    # Display the table.
    df_stats.to_csv('train_summary', sep='\t')


def get_premodel(name):
    tokenizer = T5Tokenizer.from_pretrained(name, do_lower_case=True)
    model = T5ForConditionalGeneration.from_pretrained(
        name,
        #num_labels=3,
        output_attentions=False,  # Whether the model returns attentions weights.
        output_hidden_states=False,
    )
    model.cuda()
    print("model load end!")
    return model, tokenizer


l = 'allenai/longformer-large-4096-finetuned-triviaqa'
b = 'google/bigbird-roberta-large'
p = 'google/bigbird-pegasus-large-arxiv'
T = 'google/t5-v1_1-large'

model, tokenizer = get_premodel(T)
model = nn.DataParallel(model)
data = pd.read_csv('./qa_0_all.csv')

'''
tokenize all of the sentences and map the tokens to thier word IDs.

'''
# input_ids = []
# attention_masks = []
# #sentences = data.context.values
# #qs = data.question.values
# # For every sentence...
# for _ in zip(qs, sentences):
#     encoded_dict = tokenizer.encode_plus(
#         _[0],  # Sentence pair to encode.
#         _[1],
#         add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
#         max_length=1048 + 512,  # Pad & truncate all sentences.
#         pad_to_max_length=True,
#         truncation=True,
#         return_attention_mask=True,  # Construct attn. masks.
#         return_tensors='pt',  # Return pytorch tensors.
#     )
#
#     # Add the encoded sentence to the list.
#     input_ids.append(encoded_dict['input_ids'])
#
#     # And its attention mask (simply differentiates padding from non-padding).
#     attention_masks.append(encoded_dict['attention_mask'])
#
#
step_number = len(data)
#
# # Convert the lists into tensors.
# input_ids = torch.cat(input_ids, dim=0)
# attention_masks = torch.cat(attention_masks, dim=0)
# labels = data.answer.astype(int).values - 1
# labels = torch.tensor(labels)

class myDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

        self.max_source_length = 1024+512
        self.max_target_length = 5

    def __getitem__(self, index):
        item = self.data.iloc[index]
        qs = item.question
        ans = str(item.answer)
        texts = item.context

        #q_text = q + " : " + text
        source_encoding = self.tokenizer.encode_plus(qs,
                                                     texts,
                                                     add_special_tokens=True,
                                                     max_length=self.max_source_length,
                                                     padding='max_length',
                                                     return_attention_mask=True,
                                                     return_tensors='pt',
                                                     truncation=True)
        input_ids, attention_mask = source_encoding['input_ids'], source_encoding['attention_mask']
        # 答案
        target_encoding = self.tokenizer.encode_plus(ans,
                                                     add_special_tokens=True,
                                                     max_length=self.max_target_length,
                                                     padding='max_length',
                                                     return_attention_mask=True,
                                                     truncation=True)
        labels = target_encoding.input_ids
        labels = torch.tensor(labels)

        return input_ids.squeeze(), attention_mask.squeeze(), labels

    def __len__(self):
        return len(self.data)


def shuffle_dataset(thedataset):
    dataset = thedataset

    # Calculate the number of samples to include in each set.
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    # Divide the dataset by randomly selecting samples.
    train_dataset, val_dataset = train_test_split(dataset, test_size=0.1, random_state=42, shuffle=True)
    train_dataset = train_dataset.reset_index(drop=True)
    val_dataset = val_dataset.reset_index(drop=True)

    print('{:>5,} training samples'.format(train_size))
    print('{:>5,} validation samples'.format(val_size))

    '''

    The DataLoader needs to know our batch size for training, so we specify it 
    here. For fine-tuning BERT on a specific task, the authors recommend a batch 
    size of 16 or 32.

    '''
    # batch_size = batch_size

    # # Create the DataLoaders for our training and validation sets.
    # # We'll take training samples in random order.
    # train_dataloader = DataLoader(
    #     train_dataset,  # The training samples.
    #     sampler=RandomSampler(train_dataset),  # Select batches randomly
    #     batch_size=batch_size  # Trains with this batch size.
    # )
    #
    # # For validation the order doesn't matter, so we'll just read them sequentially.
    # validation_dataloader = DataLoader(
    #     val_dataset,  # The validation samples.
    #     sampler=SequentialSampler(val_dataset),  # Pull out batches sequentially.
    #     batch_size=batch_size  # Evaluate with this batch size.
    # )
    return train_dataset, val_dataset

def loaddata(train,val,batch_size):
    train_dataloader = DataLoader(
            train,  # The training samples.
            sampler=RandomSampler(train_dataset),  # Select batches randomly
            batch_size=batch_size  # Trains with this batch size.
        )

        # For validation the order doesn't matter, so we'll just read them sequentially.
    validation_dataloader = DataLoader(
            val,  # The validation samples.
            sampler=SequentialSampler(val_dataset),  # Pull out batches sequentially.
            batch_size=batch_size  # Evaluate with this batch size.
        )
    return train_dataloader, validation_dataloader
'''

build model

'''

optimizer = AdamW(model.parameters(),
                  lr=2e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps=1e-8,  # args.adam_epsilon  - default is 1e-8.
                  weight_decay=0.2
                  )

epochs = 6
# Total number of training steps is [number of batches] x [number of epochs].
# (Note that this is not the same as the number of training samples).

total_steps = step_number * 0.9 * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,  # Default value in run_glue.py
                                            num_training_steps=total_steps)

# We'll store a number of quantities such as training and validation loss,
# validation accuracy, and timings.
training_stats = []

# Measure the total training time for the whole run.
total_t0 = time.time()

for epoch_i in range(0, epochs):
    # ========================================
    #               Training
    # ========================================

    # Perform one full pass over the training set.

    train_dataset, val_dataset = shuffle_dataset(data)
    train_Dataset = myDataset(train_dataset, tokenizer)
    val_Dataset = myDataset(val_dataset, tokenizer)
    train_dataloader, validation_dataloader = loaddata(train_Dataset, val_Dataset, batch_size=3)
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    # Measure how long the training epoch takes.
    t0 = time.time()
    # Reset the total loss for this epoch.
    total_train_loss = 0
    model.train()

    # For each batch of training data...
    for step, batch in enumerate(train_dataloader):
        print(step)
        if step % 40 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)

            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            # `batch` contains three pytorch tensors:
            #   [0]: input ids
            #   [1]: attention masks
            #   [2]: labels
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        # set grad to zero.
        model.zero_grad()
        # perform a forward pass. logits is somewhat the model outputs prior to activation.
        outputs = model(b_input_ids,
                        # token_type_ids=None,
                        attention_mask=b_input_mask,
                        labels=b_labels)
        total_train_loss += outputs.loss.sum()
        outputs.loss.sum().backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # Update parameters and take a step using the computed gradient.
        # The optimizer dictates the "update rule"--how the parameters are
        # modified based on their gradients, the learning rate, etc.
        optimizer.step()
        # Update the learning rate.
        scheduler.step()
    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_dataloader)
    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)

    print("")
    print("  avg training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(training_time))

    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.
    print("")
    print("Running Validation...")
    t0 = time.time()

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()
    # Tracking variables
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0
    # Evaluate data for one epoch
    for batch in validation_dataloader:
        # `batch` contains three pytorch tensors:
        #   [0]: input ids
        #   [1]: attention masks
        #   [2]: labels
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        # Tell pytorch not to bother with constructing the compute graph during
        # the forward pass, since this is only needed for backprop (training).
        with torch.no_grad():
            outputs = model(b_input_ids,
                            # token_type_ids=None,
                            attention_mask=b_input_mask,
                            labels=b_labels)
        # Accumulate the validation loss.

        total_eval_loss += outputs.loss.sum()
        # Move logits and labels to CPU
        # logits = outputs.logits.detach().cpu().numpy()
        # label_ids = b_labels.to('cpu').numpy()

        decode_labels = tokenizer.decode(b_labels[0], skip_special_tokens=True)
        decode_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if str(decode_labels) == str(decode_output):
            total_eval_accuracy+=1
        else:
            total_eval_accuracy+=0

        # Calculate the accuracy for this batch of test sentences, and
        # accumulate it over all batches.
        # print('lgoits is:  ', logits, '\n')

    avg_val_loss = total_eval_loss / len(validation_dataloader)
    # Report the final accuracy for this validation run.
    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    print("  Accuracy: {0:.3f}".format(avg_val_accuracy))
    # Measure how long the validation run took.
    validation_time = format_time(time.time() - t0)
    print("  Validation Loss: {0:.3f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))

    # Record all statistics from this epoch.
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )

print("")
print("Training complete!")
print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))

train_summary(training_stats)

torch.save(model.state_dict(), "model_parameter.pkl")
