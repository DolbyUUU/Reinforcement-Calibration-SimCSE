import torch
from torch import nn as nn
import numpy as np
import matplotlib.pyplot as plt
from transformers import BertModel, BertTokenizer
from tqdm import tqdm, trange
import torch.nn.functional as F
import random
import sys
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QRadioButton, QVBoxLayout, QHBoxLayout
import pickle

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




class DDPGLoss(nn.Module):
 
	def __init__(self):
		super().__init__()
	
	def forward(self, pi, rewardi, k = 50):
		return - k * pi * rewardi


def getReward(realvalue, predictvalue):
	'''
	Reward function is: -100(x - x0)^2 + 1
	'''
	reward_value = - 100 * (predictvalue - realvalue) ** 2 + 1
	if reward_value < 0:
		reward_value /= 10
	return reward_value


def getState(test_corpus):

	# setting sentence_length to obtain the fixed length input
	sentence_length = 128

	# for the test set
	split_test_corpus_id = []

	for sentence in test_corpus:
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
		split_test_corpus_id.append(token_id)

	split_test_corpus_id = torch.LongTensor(split_test_corpus_id).cuda()

	return split_test_corpus_id




class MainWindow(QWidget):
	def __init__(self):
		super().__init__()

		# read corpus
		with open("wikisent2.txt", "r", encoding = "utf8") as f:
			corpus = f.read().splitlines()

		# shuffle the corpus
		random.shuffle(corpus)
			
		# select some of the corpus as the test_corpus
		self.test_corpus = corpus[:100000]

		#---！Super Parameters Definition
		learning_rate = 1e-5 # RLHF lr (not the training lr)
		self.random_factor = 0.95 # fator of discovery
		self.device = "cuda:0"

		# initialize the critic-Network (the BERT)
		self.critic_network = PerceptualBERT().to(self.device)
		self.critic_network.load_state_dict(torch.load("fine_tune_model.pt"))

		# initialize the loss function (-Pi * Rewardi)
		self.loss_fn = DDPGLoss().to(self.device)

		# initialize the optimizer 
		self.optimizer = torch.optim.Adam(self.critic_network.parameters(),lr = learning_rate)

		# initilize the probability estimator (to estimate pi, by reading the file)
		#self.probability_estimator = {x / 100: 1 for x in range(101)}
		with open("probability_estimator.pkl", "rb") as f:
			self.probability_estimator = pickle.load(f)


		# create two editable text box
		self.sentence1 = QLineEdit()
		self.sentence2 = QLineEdit()

		# select two sentences from test set and set to the lineedit
		self.sentence1.setText(random.choice(self.test_corpus))
		self.sentence2.setText(random.choice(self.test_corpus))


		# create 5 single box
		self.radio_very_low = QRadioButton("very low (<0.2)")
		self.radio_low = QRadioButton("low (0.2-0.4)")
		self.radio_medium = QRadioButton("medium (0.4-0.6)")
		self.radio_high = QRadioButton("high (0.6-0.8)")
		self.radio_very_high = QRadioButton("very high (>0.8)")

		# create next buttom
		self.btn_next = QPushButton("Next")
		self.btn_next.clicked.connect(self.iterate)

		# create vertical layout
		vbox = QVBoxLayout()

		# add to layout
		vbox.addWidget(QLabel("Sentence 1:"))
		vbox.addWidget(self.sentence1)
		vbox.addWidget(QLabel("Sentence 2:"))
		vbox.addWidget(self.sentence2)
		vbox.addWidget(QLabel("Choose the similarity:"))
		vbox.addWidget(self.radio_very_low)
		vbox.addWidget(self.radio_low)
		vbox.addWidget(self.radio_medium)
		vbox.addWidget(self.radio_high)
		vbox.addWidget(self.radio_very_high)

		# add "next" button to the horizontal layout
		hbox = QHBoxLayout()
		hbox.addStretch()
		hbox.addWidget(self.btn_next)

		# create overall layout
		vbox.addLayout(hbox)

		# setting the layout
		self.setLayout(vbox)

		# setting widget size
		self.setWindowTitle("RLHF for Similarity")
		self.setGeometry(100, 100, 1400, 700)
		font = QFont("Times New Roman", 20)
		self.setFont(font)


	def iterate(self):

		# first, obtain the target similarity
		if self.radio_very_low.isChecked():
			target_similarity = 0.05
		elif self.radio_low.isChecked():
			target_similarity = 0.3
		elif self.radio_medium.isChecked():
			target_similarity = 0.5
		elif self.radio_high.isChecked():
			target_similarity = 0.7
		elif self.radio_very_high.isChecked():
			target_similarity = 0.95
		else:
			target_similarity = 0.5


		# then, get state
		state = getState([self.sentence1.text(), self.sentence2.text()])
		state = state.to(self.device)

		# obtain the output embedding
		_, output_embedding = self.critic_network(state)

		# obtain the output similarity
		output_similarity = torch.abs(F.cosine_similarity(output_embedding[0].view(1, -1), output_embedding[1].view(1, -1)).squeeze())

		# obtain the action (0.01 as a stage)
		if np.random.uniform() <= self.random_factor:
			action = round(output_similarity.item(), 2)
		else:
			action = random.randint(0,100) / 100

		# update the probability estimator
		self.probability_estimator[action] += 1

		# obtain pi
		pi = self.probability_estimator[action] / np.sum(list(self.probability_estimator.values()))

		# obtain the reward
		rewardi = getReward(target_similarity, output_similarity)

		# calculate the loss function
		loss = self.loss_fn(pi, rewardi)

		# update the critic network
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		# select two sentences from test set and set to the lineedit
		self.sentence1.setText(random.choice(self.test_corpus))
		self.sentence2.setText(random.choice(self.test_corpus))

	def closeEvent(self, event):

		# save probability estimator
		with open("probability_estimator.pkl", "wb") as f:
			pickle.dump(self.probability_estimator, f)

		# save model
		torch.save(self.critic_network.state_dict(), "fine_tune_model.pt")


if __name__ == '__main__':
	app = QApplication(sys.argv)
	main_window = MainWindow()
	main_window.show()
	sys.exit(app.exec_())
