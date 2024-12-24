# Import All Libraries
from transformers import BertModel, BertTokenizer
import torch
from torch import nn as nn
import numpy as np
from tqdm import tqdm, trange
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
import sys
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QRadioButton, QVBoxLayout, QHBoxLayout
import pickle
import senteval


# Load the Based BERT Weight From Pretrained
basemodel = BertModel.from_pretrained('google/bert_uncased_L-6_H-256_A-4', output_hidden_states=True)
tokenizer = BertTokenizer.from_pretrained('google/bert_uncased_L-6_H-256_A-4')


# construct our model
class PerceptualBERT(nn.Module):
    def __init__(self):
        super(PerceptualBERT, self).__init__()
        
        # Using Embedding layer weights as the word embedding
        self.embedding = basemodel.embeddings
        
        # Extract the encoder layer weights as our encoder weights
        self.encoder1 = basemodel.encoder.layer[0]
        self.encoder2 = basemodel.encoder.layer[1]
        self.encoder3 = basemodel.encoder.layer[2]
        self.encoder4 = basemodel.encoder.layer[3]
        self.encoder5 = basemodel.encoder.layer[4]
        self.output_layer = basemodel.pooler.dense
        
    def forward(self, input_sentence_pairs):
        '''
        input_sentence_pairs: shape: (batch_size = ..., seq_len = 128), i.e. batch_size = 16 means there are 8 pairs of sentence
        '''
        
        # first, calculate the embedding of each word in the sentence (batch_size, seq_len, embedding_dim)
        sentence_embedding = self.embedding(input_sentence_pairs)
        
        # then, going through each encoder (shape remains unchanged!)
        x = self.encoder1(sentence_embedding)
        x = self.encoder2(x[0])
        x = self.encoder3(x[0])
        x = self.encoder4(x[0])
        x = self.encoder5(x[0])
        x_output = self.output_layer(x[0])
        
        return sentence_embedding, torch.mean(x_output, dim = 2)


# define the loss function (force-based + InfoNCE)
class ForceBasedInfoNCE(nn.Module):
 
    def __init__(self):
        super().__init__()
    
    def force_field(self, x, threshold = 0.95):
        x = torch.abs(x)
        x[x < threshold] = (x[x < threshold] - threshold)**2
        x[x >= threshold] = - (threshold**2 / ((threshold - 1)**2) ) * (x[x >= threshold] - 1)**2
        return x
 
    def forward(self, input_embeddings, output_embeddings, ifprint = False):
        
        # first, calculate the estimated similarity, it should first calculate the sentence embeddings for each input_embeddings
        estimated_sentence_embedding = torch.mean(input_embeddings, dim = 2)
        
        # then, calculate the cosine_similarity
        estimated_cosine_similarity = F.cosine_similarity(estimated_sentence_embedding.unsqueeze(1), estimated_sentence_embedding.unsqueeze(0), dim = -1)
        
        # calculate the force field (if the element < 0, it should come closer)
        trend_move = self.force_field(estimated_cosine_similarity)
        
        # calculate the distance between all matrix
        real_distance = torch.cdist(output_embeddings, output_embeddings).reshape(output_embeddings.shape[0], output_embeddings.shape[0])

        # obtain the loss (inplace with trend move) and prevent to be zero
        trend_move[trend_move != 0.] = trend_move[trend_move != 0.] / (real_distance[trend_move != 0] + 1e-8)
        trend_move[trend_move == 0.] = real_distance[trend_move == 0.]

        if not ifprint:
            return torch.sum(trend_move)
        else:
            return trend_move, real_distance, estimated_cosine_similarity

# Read the Corpus and Preprocessing it
# read corpus
import random

with open("wikisent2.txt", "r", encoding = "utf8") as f:
    corpus = f.read().splitlines()

# shuffle the corpus
random.shuffle(corpus)
    
# select some of the corpus as the training corpus (supervised learning does not need test corpus!)
training_corpus = corpus[:1000000]

# preprocessing the corpus
split_training_corpus_id = []

# setting sentence_length to obtain the fixed length input
sentence_length = 128

for sentence in tqdm(training_corpus):
    # tokenizer
    token_body = tokenizer.tokenize(sentence)
    token = ['[CLS]'] + token_body
    
    # add padding using the original sentence
    num_repeats = (sentence_length // len(token_body))
    
    if num_repeats > 0:
        padding_list = token_body * num_repeats
        token = (token + padding_list + ['[SEP]'])[:sentence_length]
    else:
        token = token[:sentence_length]
    
    # convert to ID
    token_id = tokenizer.convert_tokens_to_ids(token)
    
    # add to training_list
    split_training_corpus_id.append(token_id)

# setting the device
device = "cuda:0"

# transfer the training corpus into tensor
tensor_split_training_corpus_id = torch.LongTensor(split_training_corpus_id).to(device)

# instance the model
model = PerceptualBERT().to(device)

# load checkpoint (if needed)
checkpoint = 0
if checkpoint:
    model.load_state_dict(torch.load("model0.pt"))
else:
    # Xavier Initialization
    #for name, param in model.named_parameters():
    #    if 'weight' in name and len(param.shape) > 1:
    #        nn.init.xavier_uniform_(param)
    pass

# construct the loss function
loss_fn = ForceBasedInfoNCE().to(device)

# define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-7)

# define the batch size
batch_size = 64

# define the iteration epoch
num_epoch = 1

# begin iteration, number of epoch means training hold training set num_epoch times
for epoch in range(num_epoch):
    # iterate over the training set in batches
    pbar = trange(0, tensor_split_training_corpus_id.shape[0], batch_size)
    for it in pbar:
        # inputing small batches of training set to the PerceptualBERT model and obtianed the original embedding and the output embedding
        original_embedding, calibration_embedding = model(
            torch.cat((tensor_split_training_corpus_id[it:it+batch_size], tensor_split_training_corpus_id[it:it+batch_size]), dim=0)
        )
        # compute the loss between original and calibration embeddings
        loss = loss_fn(original_embedding, calibration_embedding)
        # zero the gradient of optimizer (otherwise the gradient would be accumulated)
        optimizer.zero_grad(set_to_none = False)
        # compute the gradients of the loss and backward it
        loss.backward()

        # update the model parameters using the optimizer
        optimizer.step()

        # update the tqdm progress bar with the current loss
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    # print epoch number and current loss
    print(f"======Epoch{epoch}======")
    # save the current state of the model
    torch.save(model.state_dict(), f"model{epoch}.pt")