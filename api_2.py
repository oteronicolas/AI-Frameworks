from annoy import AnnoyIndex
from flask import Flask, jsonify, request
import os
import pandas as pd
from nltk import word_tokenize
from nltk.stem import SnowballStemmer
import nltk
from nltk.corpus import stopwords
import re
import pickle
import numpy as np
import json



vectorizer = pd.read_pickle("tfidf.pkl")
bag_count= pd.read_pickle("BoW_Count.pkl")

def bag_tidf(texte):
    
    
    matrix = vectorizer.transform([texte])

    return matrix[0]

def bag(texte):
    matrix = bag_count.transform([texte])

    return matrix[0]


app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
 data = request.json
 df = data.get('vec')
 k = data.get('nb_reco')
 m = data.get('method')
 print("LAAAAAAAAAAAAAAAAAAAAA",df)
 if m =="image":
     emb=df
     dim = 576
     annoy_index = AnnoyIndex(dim, 'angular')
     annoy_index.load('rec_imdb.ann')

     
 elif m == "Bag_of_words(tidf)":
     emb_sp=bag_tidf(df)
     emb=emb_sp.toarray().flatten()
     print("TAILLE",emb.shape)
     print("EMB",np.where(emb!=0))
     dim=5000
     annoy_index = AnnoyIndex(dim, 'angular')
     annoy_index.load('Tfidf.ann')
     
 elif m == "Bag_of_words":
     emb_sp=bag(df)
     emb=emb_sp.toarray().flatten()
     dim=5000
     annoy_index = AnnoyIndex(dim, 'angular')
     annoy_index.load('BoW_Count.ann')
    
 elif m =="Glove":
    print("hhhh")
    """
    # Charger le modèle
    model = TextClassificationModel(vocab_size, embedding_dim, num_class, use_pre_trained=True)
    model.load_state_dict(torch.load("TextClassificationModel_weights.pkl"))
    model.eval()  
    
    processed_text = process_text(df)

    # Charger les données associées (par exemple, embeddings GloVe)
    with open("glove_embeddings.pkl", "rb") as file:
        glove_embeddings = pickle.load(file)
        
    with torch.no_grad():
        emb = model.get_embeddings(processed_text.unsqueeze(0).to(device)).cpu().numpy()

    
    # Charger l'index Annoy
    annoy_index = AnnoyIndex(5000, 'angular')
    annoy_index.load("annoy_index.ann")
    """
 indices = annoy_index.get_nns_by_vector(emb, k)
 return jsonify({"prediction": indices})


if __name__ == "__main__":
 app.run(host='0.0.0.0', port=5000, debug=False)