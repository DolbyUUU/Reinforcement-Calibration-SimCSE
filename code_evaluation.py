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


# Load the trained model
device = "cuda:0"
model = PerceptualBERT().to(device)
model.load_state_dict(torch.load("model0.pt"))
# model.load_state_dict(torch.load("fine_tune_model.pt"))
model.eval()  # Set the model to evaluation mode


def prepare(params, samples):
    pass


def batcher(params, batch):
    # Tokenize the batch of sentences
    batch_tokens = [tokenizer.encode(sentence, add_special_tokens=True) for sentence in batch]

    # Set the fixed length for padding
    fixed_length = 128

    # Pad or truncate the sequences
    for idx, tokens in enumerate(batch_tokens):
        if len(tokens) > fixed_length:
            batch_tokens[idx] = tokens[:fixed_length]
        else:
            num_repeats = fixed_length // len(tokens)
            padding_list = tokens * num_repeats
            batch_tokens[idx] = (padding_list + tokens)[:fixed_length]

    # Convert the tokenized batch to a PyTorch tensor
    batch_tokens = torch.tensor(batch_tokens).to(device)

    # Extract the sentence embeddings from the model
    with torch.no_grad():
        sentence_embeddings, word_embeddings = model(batch_tokens)
    
    # Average the word embeddings to get sentence embeddings
    sentence_embeddings = torch.mean(word_embeddings, dim=1)
    
    # Convert the embeddings to a numpy array
    sentence_embeddings = sentence_embeddings.cpu().numpy()

    return sentence_embeddings



# Set up SentEval parameters
params = {
    'task_path': 'SentEval/data',
    'usepytorch': True,
    'kfold': 10,
}

# Create SentEval instance
se = senteval.engine.SE(params, batcher, prepare)

# Define the set of transfer tasks and run the evaluation
# transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']
transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'SICKRelatedness']
results = se.eval(transfer_tasks)

# Print results
print("Pearson:")
for task in transfer_tasks:
    print(f"{task}: {results[task]['all']['pearson']['mean'] * 100:.4f}")
print("Spearman:")
for task in transfer_tasks:
    print(f"{task}: {results[task]['all']['spearman']['mean'] * 100:.4f}")
