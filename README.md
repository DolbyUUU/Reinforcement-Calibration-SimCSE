# 🔥 Reinforcement Calibration SimCSE

Reinforcement Calibration SimCSE is a model designed for **Semantic Textual Similarity (STS)** tasks. It builds upon the **SimCSE framework** by incorporating **artificial potential fields**, **perceptual loss**, and **reinforcement learning from human feedback (RLHF)** to enhance the quality of sentence embeddings. This repository includes the code for training, fine-tuning, and evaluating the model, along with a user-friendly GUI for collecting human feedback.

---

## 🚀 Features

- **Innovative Loss Function**: Combines contrastive learning with artificial potential fields to address limitations in traditional sentence embedding methods.
- **Perceptual Loss Integration**: Minimizes "length bias" and improves semantic representation of sentence embeddings.
- **Fine-Tuning with RLHF**: Uses human feedback to fine-tune embeddings via a PyQt-based GUI.
- **Evaluation with SentEval**: Comprehensive benchmarking on STS datasets using the SentEval toolkit.

---

## 📖 Semantic Textual Similarity

Semantic similarity is a core problem in **Natural Language Processing (NLP)**, where the goal is to quantify how similar two linguistic items are in terms of meaning. It has applications in tasks like **lexical semantics**, **part-of-speech tagging**, **machine translation**, and **social media analysis**.

### Key Contributions in STS Research

1. **Skip-Thought** (Kiros et al., 2015): Trains an encoder-decoder architecture to predict surrounding sentences.
2. **InferSent** (Conneau et al., 2017): Trains a siamese BiLSTM network with max-pooling using labeled NLI data.
3. **Universal Sentence Encoder** (Cer et al., 2018): Augments unsupervised learning with NLI training using a transformer network.
4. **Reddit Conversations** (Yang et al., 2018): Uses siamese DAN and transformer networks to train on Reddit conversations.
5. **SimCSE** (Gao et al., 2021): A simple contrastive learning framework that greatly improved state-of-the-art performance on STS tasks.

---

### SentEval Toolkit

**SentEval** is a popular evaluation toolkit for sentence representations. It includes 17 downstream tasks, including **STS12-16**, **STS-B**, and **SICK-R**, which measure sentence relatedness using cosine similarity and Pearson correlation. Learn more at [SentEval GitHub](https://github.com/facebookresearch/SentEval).

For more details on SimCSE, visit the [SimCSE GitHub](https://github.com/princeton-nlp/SimCSE).

### Dataset

The `wikisent2.txt` dataset used for training can be downloaded from [Wikipedia Sentences on Kaggle](https://www.kaggle.com/datasets/mikeortman/wikipedia-sentences).

## Disclaimer

This project was developed as part of a group coursework assignment. Please use this project for reference or educational purposes only, and exercise caution if applying it to other use cases.
