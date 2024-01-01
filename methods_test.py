from nltk import word_tokenize
from nltk.stem import SnowballStemmer
import nltk
from nltk.corpus import stopwords
import re
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from annoy import AnnoyIndex
import os
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


############################    creation df etc ############# 
nltk.download('punkt')
nltk.download('stopwords')
# Download stopwords list

stop_words = set(stopwords.words('english'))



path_file = os.getcwd()
movies = pd.read_csv(path_file + '/movies_metadata.csv')
movies['overview'].fillna('', inplace=True)
movies.dropna(subset=['overview'], inplace=True)

movies = movies[['id', 'title', 'overview']]



#################   Tidf #############################
"""
vectorizer = TfidfVectorizer(stop_words=list(stop_words),max_features=5000)
matrix = vectorizer.fit_transform(movies['overview'])
#print(matrix.shape)


movies['Tfidf'] = None

dimension = 5000  
annoy_index = AnnoyIndex(dimension, 'angular')  

# Ajoutez les vecteurs à la base d'annoy
for i in range(matrix.shape[0]):
    vector = matrix.getrow(i).toarray().flatten()  # Obtenez le vecteur pour chaque ligne
    movies['Tfidf'].iloc[i]=vector
    annoy_index.add_item(i,movies['Tfidf'].iloc[i])

#print(movies['Tfidf'])

# Construisez l'index
annoy_index.build(n_trees=10)  # Réglez le nombre d'arbres selon vos besoins

# Sauvegardez l'index si nécessaire
annoy_index.save('Tfidf.ann')

with open("tfidf.pkl",'wb') as file:
    pickle.dump(vectorizer,file)
    

################  Bag of words classique  #########################

bag_count = CountVectorizer(stop_words=list(stop_words),max_features=5000)
matrix2 = bag_count.fit_transform(movies['overview'])
#print(matrix.shape)


movies['BoW Count'] = None

dimension = 5000  
annoy_index2 = AnnoyIndex(dimension, 'angular')  

print("LAAAAAAAAAAAAA",matrix2.shape[0])
# Ajoutez les vecteurs à la base d'annoy
for i in range(matrix2.shape[0]):
    vector2 = matrix2.getrow(i).toarray().flatten()  # Obtenez le vecteur pour chaque ligne
    movies['BoW Count'].iloc[i]=vector2
    annoy_index2.add_item(i,vector2)



# Construisez l'index
annoy_index2.build(n_trees=10)  # Réglez le nombre d'arbres selon vos besoins

# Sauvegardez l'index si nécessaire
annoy_index2.save('BoW_Count.ann')

with open("BoW_Count.pkl",'wb') as file2:
    pickle.dump(bag_count,file2)
    
"""
    
    
 ##########################  Glove    ####################################
    
    
    

from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import wget 
import zipfile
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from annoy import AnnoyIndex
import os

path_file = os.getcwd()
movies = pd.read_csv(path_file + '/movies_metadata.csv')
movies['overview'].fillna('', inplace=True)
movies.dropna(subset=['overview'], inplace=True)





nltk.download('punkt')
nltk.download('stopwords')
# Download stopwords list

stop_words = set(stopwords.words('english'))



filename = wget.download("https://nlp.stanford.edu/data/glove.6B.zip")

with zipfile.ZipFile(filename, 'r') as glove:
    glove.extractall()
os.remove(filename)



glove_file = ('glove.6B.100d.txt')
word2vec_glove_file = get_tmpfile("glove.6B.100d.word2vec.txt")
glove2word2vec(glove_file, word2vec_glove_file)
model = KeyedVectors.load_word2vec_format(word2vec_glove_file)



import torch


train_iter = iter(movies["overview"])
next(train_iter)

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

tokenizer = get_tokenizer("basic_english")
train_iter = movies["overview"]


def yield_tokens(data_iter):
    for text in data_iter:
        yield tokenizer(text)


vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])


import torch
import torch.nn as nn

def load_glove_embeddings(path, vocab, embedding_dim):
    # Load GloVe embeddings into a dictionary
    glove_embeddings = {}
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = torch.tensor([float(val) for val in values[1:]], dtype=torch.float32)
            glove_embeddings[word] = vector

    # Create a weights matrix for words in vocab
    weights_matrix = torch.zeros((len(vocab), embedding_dim))
    for word, idx in vocab.get_stoi().items():
        if word in glove_embeddings:
            weights_matrix[idx] = glove_embeddings[word]
        else:
            weights_matrix[idx] = torch.randn(embedding_dim)  # or torch.zeros(embedding_dim)

    return weights_matrix

embedding_dim = 100
glove_path = "glove.6B.100d.txt"
weights_matrix = load_glove_embeddings(glove_path, vocab, embedding_dim)



def process_text(text):
    return torch.tensor(vocab(tokenizer(text)), dtype=torch.int64)
process_text('here is the an example')



from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Collate Function for DataLoader
def collate_batch(batch):
    label_list, text_list, lengths = [], [], []
    for (_label, _text) in batch:
         label_list.append(_label - 1)  # Adjusting label to 0-based index
         processed_text = process_text(_text)
         text_list.append(processed_text)
         lengths.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    lengths = torch.tensor(lengths, dtype=torch.int64)
    text_list = torch.nn.utils.rnn.pad_sequence(text_list, batch_first=True)
    return text_list.to(device), label_list.to(device), lengths.to(device)

train_iter = movies["overview"]
train_loader = DataLoader(train_iter, batch_size=8, shuffle=True, collate_fn=collate_batch)



from torch import nn

class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class, use_pre_trained=True):
        super(TextClassificationModel, self).__init__()
        if use_pre_trained:
          self.embedding = nn.Embedding.from_pretrained(weights_matrix, freeze=True)
        else:
          self.embedding = nn.Embedding(vocab_size, embed_dim)

        # self.embedding.weight.requires_grad = False
        self.fc = nn.Linear(embed_dim, num_class)

    def get_embeddings(self, text):
        embedded = self.embedding(text)
        return torch.mean(embedded, dim=1)# compute the mean of the embeddings

    def forward(self, text):
        embedded = self.get_embeddings(text)
        output = self.fc(embedded)
        return  output# compute the logits i.e. the output of the linear layer
    
    

# model with pre-trained embeddings
LR = 0.005
num_class = 4
vocab_size = len(vocab)
embedding_dim = 100

model = TextClassificationModel(vocab_size, embedding_dim, num_class, True).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


print("DEBUT ANNOY")

movies['Glove'] = None

dimension = 5000  
annoy_index3 = AnnoyIndex(dimension, 'angular')  

embeddings, labels = [], []
#print("shape",train_loader.shape())
for i,texts in enumerate(train_loader):
    with torch.no_grad(): # we don't need to compute the gradients
        embeddings.append(model.get_embeddings(texts.to(device)).cpu().numpy()) # you need to send the embeddings to the cpu and convert them to numpy: .cpu().numpy()
        #labels.extend(label.cpu().numpy()) # same here
        movies['Glove'].iloc[i]=embeddings[i]
        annoy_index3.add_item(i,movies['Glove'].iloc[i])
        print(i)

# Construisez l'index
annoy_index3.build(n_trees=10)  # Réglez le nombre d'arbres selon vos besoins

# Sauvegardez l'index si nécessaire
annoy_index3.save('Glove.ann')


with open("TextClassificationModel_weights.pkl", 'wb') as file3:
    torch.save(model.state_dict(), file3)
    
    
with open("glove_embeddings.pkl", 'wb') as file:
    pickle.dump(weights_matrix, file) 

